//! aoc:u: the add-on content (DLC) of the running title, from `ProgramImage::add_ons`.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::object::KObject;

use crate::{Service, ServiceContext};

/// am ResultApplicationRecordNotFound (128-37), what ListAddOnContent returns for an offset past
/// the list.
const APPLICATION_RECORD_NOT_FOUND: ResultCode = ResultCode::new(128, 37);
/// nim:shop ResultNoPurchasedProductInfoAvailable (164-400).
const NO_PURCHASED_PRODUCT_INFO: ResultCode = ResultCode::new(164, 400);
const BASE_TITLE_ID_MASK: u64 = 0xffff_ffff_ffff_e000;
const AOC_TITLE_ID_OFFSET: u64 = 0x1000;
const AOC_ID_MASK: u64 = 0x7ff;

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    let shared = Arc::new(Shared {
        add_ons: context.program.add_ons.iter().map(|a| a.title_id).collect(),
        list_changed: Arc::new(Event::new()),
    });
    context.register("aoc:u", "aoc", move |context| AddOnContentManager {
        context: context.clone(),
        shared: shared.clone(),
    })
}

/// What every aoc:u session sees, as the C++ shares one manager.
struct Shared {
    add_ons: Vec<u64>,
    list_changed: Arc<Event>,
}

struct AddOnContentManager {
    context: Arc<ServiceContext>,
    shared: Arc<Shared>,
}

impl AddOnContentManager {
    fn caller_add_on_ids(&self, ctx: &HleRequestContext) -> impl Iterator<Item = u32> + '_ {
        let base = ctx.process().program_id() & BASE_TITLE_ID_MASK;
        self.shared
            .add_ons
            .iter()
            .filter(move |&&id| id & BASE_TITLE_ID_MASK == base)
            .map(|&id| (id & AOC_ID_MASK) as u32)
    }
}

impl Service for AddOnContentManager {
    fn name(&self) -> &str {
        "aoc:u"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            2 => {
                let count = self.caller_add_on_ids(ctx).count();
                ctx.push(count as u32);
                ResultCode::SUCCESS
            }
            3 => {
                let offset = ctx.pop::<u32>() as usize;
                let count = ctx.pop::<u32>() as usize;
                let ids: Vec<u32> = self.caller_add_on_ids(ctx).collect();
                if ids.len() < offset {
                    return Some(APPLICATION_RECORD_NOT_FOUND);
                }
                let listed = &ids[offset..offset + count.min(ids.len() - offset)];
                let bytes: Vec<u8> = listed.iter().flat_map(|id| id.to_le_bytes()).collect();
                if !bytes.is_empty()
                    && let Err(code) = ctx.write_buffer(0, &bytes)
                {
                    return Some(code);
                }
                ctx.push(listed.len() as u32);
                ResultCode::SUCCESS
            }
            5 => {
                let program = &self.context.program;
                let caller = ctx.process().program_id();
                let base = program
                    .nacp
                    .as_ref()
                    .filter(|_| caller == program.program_id)
                    .map_or(
                        (caller & BASE_TITLE_ID_MASK) + AOC_TITLE_ID_OFFSET,
                        |nacp| nacp.add_on_content_base_id(),
                    );
                ctx.push(base);
                ResultCode::SUCCESS
            }
            7 | 11 | 12 | 50 => ResultCode::SUCCESS,
            8 | 10 => {
                ctx.push_copy_object(KObject::ReadableEvent(self.shared.list_changed.clone()));
                ResultCode::SUCCESS
            }
            100 | 101 => {
                let manager = PurchaseEventManager {
                    purchased: Arc::new(Event::new()),
                };
                ctx.push_interface(self.context.handler(manager));
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}

struct PurchaseEventManager {
    purchased: Arc<Event>,
}

impl Service for PurchaseEventManager {
    fn name(&self) -> &str {
        "IPurchaseEventManager"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 | 1 => ResultCode::SUCCESS,
            2 => {
                ctx.push_copy_object(KObject::ReadableEvent(self.purchased.clone()));
                ResultCode::SUCCESS
            }
            3 | 4 => NO_PURCHASED_PRODUCT_INFO,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::settings::Settings;
    use lichii_fs::vfs::MemFile;
    use lichii_loader::program::AddOnContent;

    use super::*;
    use crate::testing::{self, Guest, PROGRAM_ID, Request};

    fn add_on(title_id: u64) -> AddOnContent {
        AddOnContent {
            title_id,
            version: 0,
            romfs: MemFile::from_bytes(vec![0u8; 16]),
        }
    }

    fn guest(with_nacp: bool) -> Guest {
        let mut image = testing::image();
        image.add_ons = vec![
            add_on(PROGRAM_ID + 0x1001),
            add_on(PROGRAM_ID + 0x1002),
            add_on(PROGRAM_ID + 0x1005),
            add_on(0x0100_0000_0002_1001),
        ];
        if with_nacp {
            image.nacp = Some(testing::nacp("Game", 1, 0x0100_0000_0000_7000));
        }
        Guest::new(image, Settings::default(), |context| {
            register(context).unwrap()
        })
    }

    #[test]
    fn counts_and_lists_the_titles_add_ons() {
        let guest = guest(false);
        let aoc = guest.service("aoc:u");
        let count = guest.call(aoc, Request::new(2).pid().arg(0u64)).ok();
        assert_eq!(count.at::<u32>(0), 3);

        let out = guest.heap;
        let list = guest
            .call(
                aoc,
                Request::new(3)
                    .pid()
                    .arg(1u32)
                    .arg(8u32)
                    .arg(0u64)
                    .output(out, 0x100),
            )
            .ok();
        assert_eq!(list.at::<u32>(0), 2);
        assert_eq!(guest.read(out, 8), [2, 0, 0, 0, 5, 0, 0, 0]);

        let past_end = guest.call(aoc, Request::new(3).pid().arg(4u32).arg(1u32).arg(0u64));
        assert_eq!(past_end.result, APPLICATION_RECORD_NOT_FOUND);
    }

    #[test]
    fn base_id_comes_from_the_nacp_or_the_program_id() {
        let guest = guest(true);
        let aoc = guest.service("aoc:u");
        let base = guest.call(aoc, Request::new(5).pid().arg(0u64)).ok();
        assert_eq!(base.at::<u64>(0), 0x0100_0000_0000_7000);
        drop(guest);

        let guest = self::guest(false);
        let aoc = guest.service("aoc:u");
        let base = guest.call(aoc, Request::new(5).pid().arg(0u64)).ok();
        assert_eq!(base.at::<u64>(0), PROGRAM_ID + 0x1000);
    }

    #[test]
    fn events_and_the_purchase_manager() {
        let guest = guest(false);
        let aoc = guest.service("aoc:u");
        guest
            .call(aoc, Request::new(7).pid().arg(0u32).arg(0u64))
            .ok();
        let event = guest.call(aoc, Request::new(8)).ok().handle();
        assert!(!guest.is_signaled(event));
        guest
            .call(aoc, Request::new(10).pid().arg(0u64))
            .ok()
            .handle();

        let manager = guest.call(aoc, Request::new(100)).ok().interface();
        guest.call(manager, Request::new(2)).ok().handle();
        let pop = guest.call(manager, Request::new(3));
        assert_eq!(pop.result, NO_PURCHASED_PRODUCT_INFO);
    }
}
