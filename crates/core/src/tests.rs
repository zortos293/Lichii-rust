use std::path::PathBuf;
use std::time::Duration;

use super::*;

const SVC_OUTPUT_DEBUG_STRING: u32 = 0xd400_04e1;
const SVC_EXIT_PROCESS: u32 = 0xd400_00e1;
const MESSAGE: &[u8] = b"hello from the guest";

fn nso(text: &[u8]) -> Vec<u8> {
    let mut file = vec![0u8; 0x100];
    file[..4].copy_from_slice(b"NSO0");
    let put = |file: &mut Vec<u8>, at: usize, value: u32| {
        file[at..at + 4].copy_from_slice(&value.to_le_bytes());
    };
    let segments = [text.len() as u32, 0x10, 0x10];
    let mut file_offset = 0x100u32;
    for (index, size) in segments.into_iter().enumerate() {
        put(&mut file, 0x10 + index * 0x10, file_offset);
        put(&mut file, 0x14 + index * 0x10, index as u32 * 0x1000);
        put(&mut file, 0x18 + index * 0x10, size);
        put(&mut file, 0x60 + index * 4, size);
        file_offset += size;
    }
    file.extend_from_slice(text);
    file.resize(file.len() + 0x20, 0);
    file
}

fn exit_program() -> Vec<u8> {
    let adr_x0_message = 0x1000_0000 | (4 << 5);
    let mov_x1_len = 0xd280_0001 | ((MESSAGE.len() as u32) << 5);
    let mut text: Vec<u8> = [
        adr_x0_message,
        mov_x1_len,
        SVC_OUTPUT_DEBUG_STRING,
        SVC_EXIT_PROCESS,
    ]
    .iter()
    .flat_map(|word: &u32| word.to_le_bytes())
    .collect();
    text.extend_from_slice(MESSAGE);
    text.resize(0x40, 0);
    text
}

struct TempFile(PathBuf);

impl TempFile {
    fn new(name: &str, bytes: &[u8]) -> Self {
        let path = std::env::temp_dir().join(format!("lichii-core-{}-{name}", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        Self(path)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn settings(console: ConsoleGeneration) -> Arc<Settings> {
    let mut settings = Settings::default();
    settings.system.preferred_console = console;
    settings.system.rng_seed = Some(1);
    Arc::new(settings)
}

#[test]
fn boots_an_nso_and_runs_it_to_exit() {
    for console in [ConsoleGeneration::Switch1, ConsoleGeneration::Switch2] {
        let file = TempFile::new("exit.nso", &nso(&exit_program()));
        let mut system =
            System::boot(settings(console), &file.0, NullRasterizer::factory()).unwrap();
        assert_eq!(system.console(), console);
        assert_eq!(system.kernel().core_count(), console.core_count());
        assert!(system.gpu().nx2_memory().is_none());
        assert_eq!(system.wait_for_exit(Some(Duration::ZERO)), None);
        system.run().unwrap();
        assert_eq!(
            system.wait_for_exit(Some(Duration::from_secs(10))),
            Some(ExitReason::Exited)
        );
        system.stop();
        system.stop();
        assert!(matches!(system.run(), Err(SystemError::Stopped)));
    }
}

#[test]
fn pause_freezes_guest_time() {
    let file = TempFile::new(
        "pause.nso",
        &nso(&[0x14000000u32.to_le_bytes(), [0; 4]].concat()),
    );
    let mut system = System::boot(
        settings(ConsoleGeneration::Switch1),
        &file.0,
        NullRasterizer::factory(),
    )
    .unwrap();
    system.run().unwrap();
    assert_eq!(system.wait_for_exit(Some(Duration::from_millis(20))), None);
    system.pause();
    assert!(system.is_paused());
    assert!(system.kernel().is_paused());
    let frozen = system.timing().global_time_ns();
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(system.timing().global_time_ns(), frozen);
    system.run().unwrap();
    assert!(!system.is_paused());
    std::thread::sleep(Duration::from_millis(5));
    assert!(system.timing().global_time_ns() > frozen);
}

#[test]
fn missing_title_is_a_loader_error() {
    let missing = std::env::temp_dir().join("lichii-core-no-such-title.nso");
    assert!(matches!(
        System::boot(
            settings(ConsoleGeneration::Switch1),
            &missing,
            NullRasterizer::factory()
        ),
        Err(SystemError::Loader(_))
    ));
}
