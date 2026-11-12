//! `appletOE` itself, `IApplicationProxy` and the Switch 2 graphics scale controller.

use std::sync::Arc;

use lichii_common::ResultCode;
use lichii_kernel::ipc::HleRequestContext;

use super::Applet;
use super::controllers::{
    AudioController, CommonStateGetter, DisplayController, Interface, SelfController,
    WindowController,
};
use super::functions::{ApplicationFunctions, LibraryAppletCreator};
use crate::dispatch::push_nx2_stub;
use crate::{Service, ServiceContext};

pub(crate) struct ApplicationProxyService {
    pub context: Arc<ServiceContext>,
    pub applet: Arc<Applet>,
}

impl Service for ApplicationProxyService {
    fn name(&self) -> &str {
        "appletOE"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 | 10 => {
                ctx.push_interface(self.context.handler(ApplicationProxy {
                    context: self.context.clone(),
                    applet: self.applet.clone(),
                }));
                ResultCode::SUCCESS
            }
            2000 => push_nx2_stub(&self.context, ctx, "appletOE"),
            _ => return None,
        })
    }
}

struct ApplicationProxy {
    context: Arc<ServiceContext>,
    applet: Arc<Applet>,
}

impl Service for ApplicationProxy {
    fn name(&self) -> &str {
        "IApplicationProxy"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let (context, applet) = (self.context.clone(), self.applet.clone());
        let handler = match ctx.command() {
            0 => context.handler(CommonStateGetter {
                context: context.clone(),
                applet,
            }),
            1 => context.handler(SelfController {
                context: context.clone(),
                applet,
            }),
            2 => context.handler(WindowController {
                context: context.clone(),
            }),
            3 => context.handler(AudioController { applet }),
            4 => context.handler(DisplayController { applet }),
            10 => context.handler(Interface("IProcessWindingController")),
            11 => context.handler(LibraryAppletCreator {
                context: context.clone(),
            }),
            20 => context.handler(ApplicationFunctions {
                context: context.clone(),
                applet,
            }),
            40 | 50 | 70 | 80 | 90 => {
                return Some(push_nx2_stub(&context, ctx, "IApplicationProxy"));
            }
            60 => context.handler(GraphicsScaleController),
            1000 => context.handler(Interface("IDebugFunctions")),
            _ => return None,
        };
        ctx.push_interface(handler);
        Some(ResultCode::SUCCESS)
    }
}

/// Switch 2 `IApplicationProxy` 60 (the name is a guess). `nn::oe::GetCurrentGraphicsScale` reads
/// command 10 and accepts 10, 20 or 30 (NX2_NOTES row 15).
struct GraphicsScaleController;

impl Service for GraphicsScaleController {
    fn name(&self) -> &str {
        "nx2:IGraphicsScaleController"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            10 => {
                ctx.push([10u32, 0, 0]);
                ResultCode::SUCCESS
            }
            1050 => ResultCode::SUCCESS,
            _ => return None,
        })
    }
}
