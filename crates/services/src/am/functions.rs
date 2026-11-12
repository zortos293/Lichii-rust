//! `IApplicationFunctions` and `ILibraryAppletCreator`.

use std::sync::Arc;

use lichii_common::settings::Language;
use lichii_common::{ConsoleGeneration, ResultCode};
use lichii_kernel::ipc::HleRequestContext;
use parking_lot::Mutex;

use super::storage::Storage;
use super::{Applet, NO_DATA_IN_CHANNEL, push_bool, push_event};
use crate::dispatch::push_nx2_stub;
use crate::{Service, ServiceContext};

/// Eden's `ResultUnknown`.
const UNKNOWN: ResultCode = ResultCode(u32::MAX);

/// `nn::am::service::LaunchParameterKind::UserChannel`.
const USER_CHANNEL: u32 = 1;

pub(super) struct ApplicationFunctions {
    pub context: Arc<ServiceContext>,
    pub applet: Arc<Applet>,
}

impl Service for ApplicationFunctions {
    fn name(&self) -> &str {
        "IApplicationFunctions"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let (context, applet) = (&self.context, &self.applet);
        let nacp = context.program.nacp.as_ref();
        Some(match ctx.command() {
            1 => {
                let kind: u32 = ctx.pop();
                let data = if kind == USER_CHANNEL {
                    applet.state.lock().user_channel.pop()
                } else {
                    None
                };
                match data {
                    Some(data) => push_storage(context, ctx, data),
                    None => {
                        tracing::warn!(kind, "no launch parameter");
                        NO_DATA_IN_CHANNEL
                    }
                }
            }
            20 => {
                ctx.push(0u64);
                ResultCode::SUCCESS
            }
            21 => {
                let supported = nacp.map_or(0, |nacp| nacp.supported_languages());
                let code = desired_language(context.settings.system.language, supported);
                ctx.push(language_code(code));
                ResultCode::SUCCESS
            }
            22 => {
                let result: u32 = ctx.pop();
                tracing::info!(
                    result = format_args!(
                        "{:04}-{:04}",
                        2000 + (result & 0x1ff),
                        (result >> 9) & 0x1fff
                    ),
                    "application set its terminate result"
                );
                ResultCode::SUCCESS
            }
            23 => {
                let version =
                    nacp.map_or_else(|| "1.0.0".to_owned(), |nacp| nacp.display_version());
                let mut out = [0u8; 16];
                let len = version.len().min(15);
                out[..len].copy_from_slice(&version.as_bytes()[..len]);
                ctx.push(out);
                ResultCode::SUCCESS
            }
            25 => {
                ctx.push(0u64);
                ResultCode::SUCCESS
            }
            26 => {
                ctx.push([
                    nacp.map_or(0, |nacp| nacp.user_account_save_data_size()),
                    nacp.map_or(0, |nacp| nacp.user_account_save_data_journal_size()),
                ]);
                ResultCode::SUCCESS
            }
            27 => {
                ctx.push(1u32);
                ctx.push(0u64);
                ResultCode::SUCCESS
            }
            28 => {
                ctx.push([0xfff_ffffu64, 0xfff_ffff]);
                ResultCode::SUCCESS
            }
            30..=33 => {
                let blocked = ctx.command().is_multiple_of(2);
                let mut state = applet.state.lock();
                state.home_button_short_blocked = blocked;
                state.home_button_long_blocked = blocked;
                ResultCode::SUCCESS
            }
            37 | 500 => {
                crate::dispatch::push_stub_zeroes(ctx);
                push_event(ctx, &applet.stub_event)
            }
            40 => push_bool(ctx, true),
            50 => {
                ctx.push(pseudo_device_id(context.program.pseudo_device_seed));
                ResultCode::SUCCESS
            }
            60 | 66 | 67 | 90 | 100..=102 | 1001 => ResultCode::SUCCESS,
            65 => push_bool(ctx, false),
            110 | 111 => {
                ctx.push(0i32);
                ResultCode::SUCCESS
            }
            121 => {
                applet.state.lock().user_channel.clear();
                ResultCode::SUCCESS
            }
            123 => {
                ctx.push(-1i32);
                ResultCode::SUCCESS
            }
            130 => push_event(ctx, &applet.gpu_error_event),
            140 => push_event(ctx, &applet.friend_invitation_event),
            141 => match applet.state.lock().friend_invitations.pop() {
                Some(data) => push_storage(context, ctx, data),
                None => NO_DATA_IN_CHANNEL,
            },
            150 => push_event(ctx, &applet.notification_event),
            160 => push_event(ctx, &applet.health_warning_event),
            210 => match context.program.generation {
                ConsoleGeneration::Switch2 => push_event(ctx, &applet.operation_mode_event),
                ConsoleGeneration::Switch1 => push_event(ctx, &applet.stub_event),
            },
            151 | 300 | 1000 => push_nx2_stub(context, ctx, "IApplicationFunctions"),
            320 => {
                ctx.push(0u16);
                ResultCode::SUCCESS
            }
            321 | 330 => {
                ctx.push(0u8);
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}

pub(super) struct LibraryAppletCreator {
    pub context: Arc<ServiceContext>,
}

impl Service for LibraryAppletCreator {
    fn name(&self) -> &str {
        "ILibraryAppletCreator"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            10 => {
                let size: i64 = ctx.pop();
                if size <= 0 {
                    return Some(UNKNOWN);
                }
                push_storage(&self.context, ctx, vec![0; size as usize])
            }
            _ => return None,
        })
    }
}

fn push_storage(
    context: &Arc<ServiceContext>,
    ctx: &mut HleRequestContext,
    data: Vec<u8>,
) -> ResultCode {
    ctx.push_interface(context.handler(Storage {
        context: context.clone(),
        data: Arc::new(Mutex::new(data)),
    }));
    ResultCode::SUCCESS
}

/// `nn::ns::ApplicationLanguage`: the bit index in the NACP's supported-language mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ApplicationLanguage {
    AmericanEnglish = 0,
    BritishEnglish = 1,
    Japanese = 2,
    French = 3,
    German = 4,
    LatinAmericanSpanish = 5,
    Spanish = 6,
    Italian = 7,
    Dutch = 8,
    CanadianFrench = 9,
    Portuguese = 10,
    Russian = 11,
    Korean = 12,
    TraditionalChinese = 13,
    SimplifiedChinese = 14,
    BrazilianPortuguese = 15,
}

const APPLICATION_LANGUAGES: [ApplicationLanguage; 16] = {
    use ApplicationLanguage::*;
    [
        AmericanEnglish,
        BritishEnglish,
        Japanese,
        French,
        German,
        LatinAmericanSpanish,
        Spanish,
        Italian,
        Dutch,
        CanadianFrench,
        Portuguese,
        Russian,
        Korean,
        TraditionalChinese,
        SimplifiedChinese,
        BrazilianPortuguese,
    ]
};

/// The system language if the title supports it (a mask of 0 supports everything), then American
/// and British English, then the first language the title lists.
fn desired_language(system: Language, supported: u32) -> ApplicationLanguage {
    use ApplicationLanguage as A;
    let wanted = match system {
        Language::Japanese => A::Japanese,
        Language::AmericanEnglish => A::AmericanEnglish,
        Language::French => A::French,
        Language::German => A::German,
        Language::Italian => A::Italian,
        Language::Spanish => A::Spanish,
        Language::Chinese | Language::SimplifiedChinese => A::SimplifiedChinese,
        Language::Korean => A::Korean,
        Language::Dutch => A::Dutch,
        Language::Portuguese => A::Portuguese,
        Language::Russian => A::Russian,
        Language::Taiwanese | Language::TraditionalChinese => A::TraditionalChinese,
        Language::BritishEnglish => A::BritishEnglish,
        Language::CanadianFrench => A::CanadianFrench,
        Language::LatinAmericanSpanish => A::LatinAmericanSpanish,
        Language::BrazilianPortuguese => A::BrazilianPortuguese,
    };
    let supports = |language: A| supported == 0 || supported & (1 << language as u32) != 0;
    [wanted, A::AmericanEnglish, A::BritishEnglish]
        .into_iter()
        .chain(APPLICATION_LANGUAGES)
        .find(|&language| supports(language))
        .unwrap_or(wanted)
}

/// The settings language code (`en-US`, `ja`, ...) packed into a u64 as `GetDesiredLanguage`
/// returns it.
fn language_code(language: ApplicationLanguage) -> u64 {
    use ApplicationLanguage as A;
    let code: &[u8] = match language {
        A::AmericanEnglish => b"en-US",
        A::BritishEnglish => b"en-GB",
        A::Japanese => b"ja",
        A::French => b"fr",
        A::German => b"de",
        A::LatinAmericanSpanish => b"es-419",
        A::Spanish => b"es",
        A::Italian => b"it",
        A::Dutch => b"nl",
        A::CanadianFrench => b"fr-CA",
        A::Portuguese => b"pt",
        A::Russian => b"ru",
        A::Korean => b"ko",
        A::TraditionalChinese => b"zh-Hant",
        A::SimplifiedChinese => b"zh-Hans",
        A::BrazilianPortuguese => b"pt-BR",
    };
    let mut bytes = [0u8; 8];
    bytes[..code.len()].copy_from_slice(code);
    u64::from_le_bytes(bytes)
}

/// A stable per-title UUID (version 5 layout) from the title's pseudo device seed.
pub(super) fn pseudo_device_id(seed: u64) -> [u8; 16] {
    let digest = lichii_crypto::sha256(&seed.to_le_bytes());
    let mut id: [u8; 16] = digest[..16].try_into().unwrap();
    id[6] = (id[6] & 0x0f) | 0x50;
    id[8] = (id[8] & 0x3f) | 0x80;
    id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn desired_language_falls_back_to_english_then_the_first_supported() {
        use ApplicationLanguage as A;
        assert_eq!(desired_language(Language::Japanese, 0), A::Japanese);
        assert_eq!(desired_language(Language::Japanese, 1 << 2), A::Japanese);
        assert_eq!(
            desired_language(Language::French, 1 << 1 | 1 << 4),
            A::BritishEnglish
        );
        assert_eq!(desired_language(Language::French, 1 << 12), A::Korean);
        assert_eq!(
            language_code(A::AmericanEnglish),
            u64::from_le_bytes(*b"en-US\0\0\0")
        );
        assert_eq!(
            language_code(A::LatinAmericanSpanish),
            u64::from_le_bytes(*b"es-419\0\0")
        );
    }

    #[test]
    fn pseudo_device_id_is_a_stable_v5_uuid() {
        let id = pseudo_device_id(0x0100_0000_0000_1000);
        assert_eq!(id, pseudo_device_id(0x0100_0000_0000_1000));
        assert_ne!(id, pseudo_device_id(0x0100_0000_0000_2000));
        assert_eq!(id[6] >> 4, 5);
        assert_eq!(id[8] >> 6, 0b10);
    }
}
