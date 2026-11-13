//! Command line: `lichii [OPTIONS] [GAME]`.

use std::ffi::OsString;
use std::path::PathBuf;

pub const USAGE: &str = "\
Usage: lichii [OPTIONS] [GAME]

Boots GAME (an NSP, XCI, NCA, NUP, NRO, NSO or a loose ExeFS directory) when given, otherwise
opens the game list.

Options:
  -f, --fullscreen     Start in fullscreen
  -c, --config <FILE>  Use FILE instead of the profile's config.toml
  -h, --help           Print this help
  -V, --version        Print the version";

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cli {
    pub game: Option<PathBuf>,
    pub fullscreen: bool,
    pub config: Option<PathBuf>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    Run(Cli),
    Help,
    Version,
}

/// Parses the arguments after the program name.
pub fn parse(args: impl IntoIterator<Item = OsString>) -> Result<Command, String> {
    let mut cli = Cli::default();
    let mut args = args.into_iter();
    let mut positional_only = false;
    while let Some(arg) = args.next() {
        let text = arg.to_string_lossy();
        if positional_only || !text.starts_with('-') || text == "-" {
            if cli.game.replace(PathBuf::from(&arg)).is_some() {
                return Err("only one game can be booted".into());
            }
            continue;
        }
        match text.as_ref() {
            "--" => positional_only = true,
            "-h" | "--help" => return Ok(Command::Help),
            "-V" | "--version" => return Ok(Command::Version),
            "-f" | "--fullscreen" => cli.fullscreen = true,
            "-c" | "--config" => {
                let file = args.next().ok_or("--config needs a file")?;
                cli.config = Some(PathBuf::from(file));
            }
            other => return Err(format!("unknown option {other}")),
        }
    }
    Ok(Command::Run(cli))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(args: &[&str]) -> Result<Command, String> {
        parse(args.iter().map(OsString::from))
    }

    #[test]
    fn no_arguments_opens_the_game_list() {
        assert_eq!(run(&[]), Ok(Command::Run(Cli::default())));
    }

    #[test]
    fn a_path_boots_it() {
        let Ok(Command::Run(cli)) = run(&["-f", "/games/zelda.nsp"]) else {
            panic!()
        };
        assert_eq!(cli.game, Some(PathBuf::from("/games/zelda.nsp")));
        assert!(cli.fullscreen);
    }

    #[test]
    fn options() {
        assert_eq!(run(&["--help", "x"]), Ok(Command::Help));
        assert_eq!(run(&["-V"]), Ok(Command::Version));
        let Ok(Command::Run(cli)) = run(&["--config", "my.toml", "--", "-odd.nro"]) else {
            panic!()
        };
        assert_eq!(cli.config, Some(PathBuf::from("my.toml")));
        assert_eq!(cli.game, Some(PathBuf::from("-odd.nro")));
    }

    #[test]
    fn errors() {
        assert!(run(&["--bogus"]).is_err());
        assert!(run(&["--config"]).is_err());
        assert!(run(&["a.nsp", "b.nsp"]).is_err());
    }
}
