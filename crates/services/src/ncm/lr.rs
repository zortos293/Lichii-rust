//! lr: location resolvers. Paths of the running title and its add-ons resolve to host paths
//! unless a title redirected them.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use lichii_common::ResultCode;
use lichii_kernel::ipc::HleRequestContext;

use crate::{Service, ServiceContext};

pub(super) const PROGRAM_NOT_FOUND: ResultCode = ResultCode::new(8, 2);
pub(super) const DATA_NOT_FOUND: ResultCode = ResultCode::new(8, 3);
pub(super) const HTML_DOCUMENT_NOT_FOUND: ResultCode = ResultCode::new(8, 6);
pub(super) const ADD_ON_CONTENT_NOT_FOUND: ResultCode = ResultCode::new(8, 7);
const CONTROL_NOT_FOUND: ResultCode = ResultCode::new(8, 8);
const LEGAL_INFORMATION_NOT_FOUND: ResultCode = ResultCode::new(8, 9);
/// `lr::Path`: a NUL-padded string of 0x300 bytes.
const PATH_SIZE: usize = 0x300;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Kind {
    Program,
    Control,
    HtmlDocument,
    LegalInformation,
    Data,
}

impl Kind {
    fn not_found(self) -> ResultCode {
        match self {
            Kind::Program => PROGRAM_NOT_FOUND,
            Kind::Control => CONTROL_NOT_FOUND,
            Kind::HtmlDocument => HTML_DOCUMENT_NOT_FOUND,
            Kind::LegalInformation => LEGAL_INFORMATION_NOT_FOUND,
            Kind::Data => DATA_NOT_FOUND,
        }
    }

    fn file(self) -> &'static str {
        match self {
            Kind::Program => "program",
            Kind::Control => "control",
            Kind::HtmlDocument => "htmldocument",
            Kind::LegalInformation => "legalinformation",
            Kind::Data => "data",
        }
    }
}

/// The host path of the running title's or an add-on's content, or `None` for other titles.
fn content_path(context: &ServiceContext, kind: Kind, id: u64) -> Option<String> {
    let program = &context.program;
    let known = match kind {
        Kind::Data => program.add_ons.iter().any(|a| a.title_id == id),
        _ => id == program.program_id,
    };
    known.then(|| format!("@Host:/{id:016X}/{}", kind.file()))
}

/// Paths a title registered or redirected.
#[derive(Default)]
struct Paths(Mutex<HashMap<(Kind, u64), Vec<u8>>>);

impl Paths {
    fn set(&self, ctx: &mut HleRequestContext, kind: Kind) -> ResultCode {
        let id: u64 = ctx.pop();
        match ctx.read_buffer(0) {
            Ok(path) => {
                self.0.lock().unwrap().insert((kind, id), path);
                ResultCode::SUCCESS
            }
            Err(code) => code,
        }
    }

    fn erase(&self, ctx: &mut HleRequestContext, kind: Kind) -> ResultCode {
        let id: u64 = ctx.pop();
        self.0.lock().unwrap().remove(&(kind, id));
        ResultCode::SUCCESS
    }

    /// Writes the redirected path of `id`, else `fallback`'s, into output buffer 0.
    fn resolve(
        &self,
        ctx: &mut HleRequestContext,
        kind: Kind,
        fallback: impl FnOnce(u64) -> Option<String>,
    ) -> ResultCode {
        let id: u64 = ctx.pop();
        let mut path = [0u8; PATH_SIZE];
        let redirected = self.0.lock().unwrap().get(&(kind, id)).cloned();
        let Some(source) = redirected.or_else(|| fallback(id).map(String::into_bytes)) else {
            return kind.not_found();
        };
        let len = source
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(source.len())
            .min(PATH_SIZE - 1);
        path[..len].copy_from_slice(&source[..len]);
        match ctx.write_buffer(0, &path) {
            Ok(_) => ResultCode::SUCCESS,
            Err(code) => code,
        }
    }
}

pub(super) struct LocationResolverManager {
    pub context: Arc<ServiceContext>,
}

impl Service for LocationResolverManager {
    fn name(&self) -> &str {
        "lr"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let context = self.context.clone();
        let handler = match ctx.command() {
            0 => self.context.handler(LocationResolver {
                context,
                redirects: Paths::default(),
            }),
            1 => self.context.handler(RegisteredLocationResolver {
                paths: Paths::default(),
            }),
            2 => return Some(ResultCode::SUCCESS),
            3 => self
                .context
                .handler(AddOnContentLocationResolver { context }),
            _ => return None,
        };
        ctx.push_interface(handler);
        Some(ResultCode::SUCCESS)
    }
}

/// ILocationResolver of one storage.
struct LocationResolver {
    context: Arc<ServiceContext>,
    redirects: Paths,
}

impl Service for LocationResolver {
    fn name(&self) -> &str {
        "ILocationResolver"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let context = &self.context;
        let resolve = |ctx: &mut HleRequestContext, kind| {
            self.redirects
                .resolve(ctx, kind, |id| content_path(context, kind, id))
        };
        Some(match ctx.command() {
            0 | 16 => resolve(ctx, Kind::Program),
            2 => resolve(ctx, Kind::Control),
            3 => resolve(ctx, Kind::HtmlDocument),
            4 => resolve(ctx, Kind::Data),
            7 => resolve(ctx, Kind::LegalInformation),
            1 | 10 | 17 | 18 => self.redirects.set(ctx, Kind::Program),
            5 => self.redirects.set(ctx, Kind::Control),
            6 => self.redirects.set(ctx, Kind::HtmlDocument),
            8 => self.redirects.set(ctx, Kind::LegalInformation),
            9 => ResultCode::SUCCESS,
            11 => {
                self.redirects.0.lock().unwrap().clear();
                ResultCode::SUCCESS
            }
            12 | 19 => self.redirects.erase(ctx, Kind::Program),
            13 => self.redirects.erase(ctx, Kind::Control),
            14 => self.redirects.erase(ctx, Kind::HtmlDocument),
            15 => self.redirects.erase(ctx, Kind::LegalInformation),
            _ => return None,
        })
    }
}

/// IRegisteredLocationResolver: only what was registered or redirected resolves.
struct RegisteredLocationResolver {
    paths: Paths,
}

impl Service for RegisteredLocationResolver {
    fn name(&self) -> &str {
        "IRegisteredLocationResolver"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => self.paths.resolve(ctx, Kind::Program, |_| None),
            1 | 3 => self.paths.set(ctx, Kind::Program),
            2 => self.paths.erase(ctx, Kind::Program),
            4 => self.paths.resolve(ctx, Kind::HtmlDocument, |_| None),
            5 | 7 => self.paths.set(ctx, Kind::HtmlDocument),
            6 => self.paths.erase(ctx, Kind::HtmlDocument),
            8 | 9 => {
                self.paths.0.lock().unwrap().clear();
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}

/// IAddOnContentLocationResolver over the title's add-ons.
struct AddOnContentLocationResolver {
    context: Arc<ServiceContext>,
}

impl Service for AddOnContentLocationResolver {
    fn name(&self) -> &str {
        "IAddOnContentLocationResolver"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                let resolved = Paths::default().resolve(ctx, Kind::Data, |id| {
                    content_path(&self.context, Kind::Data, id)
                });
                if resolved == DATA_NOT_FOUND {
                    ADD_ON_CONTENT_NOT_FOUND
                } else {
                    resolved
                }
            }
            1..=4 => ResultCode::SUCCESS,
            _ => return None,
        })
    }
}
