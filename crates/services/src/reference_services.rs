//! The services the reference registers, so boot can name those a title may open that Lichii
//! has not registered: a title that opens one waits in `sm` forever.

use crate::ServiceContext;

const REFERENCE_SERVICES: [&str; 265] = [
    "acc:aa", "acc:e", "acc:e:u1", "acc:e:u2", "acc:su", "acc:u0", "acc:u1", "ahid:cd", "ahid:hdr",
    "aoc:u", "apm", "apm:am", "apm:p", "apm:sys", "arp:r", "arp:w", "aud:a", "aud:d", "aud:u",
    "audctl", "auddev", "auddmg:u", "audin:a", "audin:d", "audin:u", "audout:a", "audout:d",
    "audout:u", "audrec:a", "audrec:d", "audrec:u", "audren:a", "audren:d", "audren:u", "audsmx:u",
    "bcat:a", "bcat:m", "bcat:s", "bcat:u", "bgtc:sc", "bgtc:t", "bpc", "bpc:ams", "bpc:b",
    "bpc:c", "bpc:r", "bpc:w", "bsd:nu", "bsdcfg", "bt", "btdrv", "btm", "btm:dbg", "btm:sys",
    "btm:u", "caps:a", "caps:c", "caps:dc", "caps:sc", "caps:ss", "caps:su", "caps:u", "chat:u",
    "clkrst", "clkrst:a", "clkrst:i", "csrng", "dauth:0", "dns:priv", "ectx:aw", "ectx:r",
    "ectx:w", "erpt:c", "erpt:r", "es", "eth:nd", "ethc:c", "ethc:i", "eupld:c", "eupld:r",
    "fatal:p", "fatal:u", "fgm", "fgm:0", "fgm:9", "fgm:dbg", "friend:a", "friend:m", "friend:s",
    "friend:u", "friend:v", "fsp-ldr", "fsp-pr", "fsp-srv", "gpio", "grc:c", "grc:d", "hid",
    "hid:dbg", "hid:sys", "hid:tmp", "hidbus", "hshl:set", "hshl:sys", "htc:tenv", "hwopus", "i2c",
    "idle:sys", "ifcfg", "ins:r", "ins:s", "irs", "irs:sys", "jit:u", "lbl", "ldn:m", "ldn:s",
    "ldn:u", "ldr:dmnt", "ldr:pm", "ldr:ro", "ldr:shel", "lm", "lm:get", "lp2p:m", "lr",
    "mig:user", "mii:e", "mii:u", "miiimg", "mm:u", "mnpp:app", "mnpp:sys", "mnpp:web", "ncm",
    "ncm:v", "nd:app", "nd:sys", "ndrm:la", "ndrm:lu", "news:a", "news:c", "news:m", "news:p",
    "news:v", "nfc:am", "nfc:mf:u", "nfc:sys", "nfc:user", "nfp:dbg", "nfp:sys", "nfp:user",
    "ngc:u", "ngct:s", "ngct:u", "nifm:a", "nifm:s", "nifm:u", "nim", "nim:eca", "nim:ecas",
    "nim:shp", "notif:a", "notif:s", "npns:s", "npns:u", "ns:am2", "ns:dev", "ns:ec", "ns:rid",
    "ns:ro", "ns:rt", "ns:su", "ns:vm", "ns:web", "nsd:a", "nsd:u", "ntc", "nvdbg:d", "nvdrv",
    "nvdrv:a", "nvdrv:s", "nvdrv:t", "nvgem:c", "nvgem:cd", "nvmemp", "olsc:s", "olsc:u", "omm",
    "ovln:rcv", "ovln:snd", "pcie", "pcie:log", "pctl", "pctl:a", "pctl:r", "pctl:s", "pcv",
    "pcv:arb", "pcv:imm", "pdm:ntfy", "pdm:qry", "pl:s", "pl:u", "pm:bm", "pm:dmnt", "pm:info",
    "pm:shell", "powctl", "prepo:a", "prepo:a2", "prepo:m", "prepo:s", "prepo:u", "psc:c", "psc:l",
    "psc:m", "psm", "psm:manu", "ro:1", "ro:dmnt", "set", "set:cal", "set:fd", "set:sys",
    "sfdnsres", "spbg:sp", "spl:", "spl:es", "spl:fs", "spl:manu", "spl:mig", "spl:ssl", "spsm",
    "ssl", "ssl:s", "stpl:sys", "stpl:u", "time:a", "time:al", "time:m", "time:p", "time:r",
    "time:su", "time:u", "ts", "usb:ds", "usb:hs", "usb:obsv", "usb:pd", "usb:pd:c", "usb:pd:m",
    "usb:pm", "usb:qdb", "vi:m", "vi:s", "wlan:dtc", "wlan:lcl", "wlan:lg", "wlan:lga", "wlan:nd",
    "wlan:p", "wlan:sg", "wlan:soc", "xcd:sys",
];

/// The reference's services the title's NPDM lets it open that nobody registered.
pub(crate) fn missing(context: &ServiceContext) -> Vec<&'static str> {
    let manager = context.kernel.service_manager();
    REFERENCE_SERVICES
        .into_iter()
        .filter(|name| context.program.services.allows_client(name) && !manager.is_registered(name))
        .collect()
}

#[cfg(test)]
mod tests {
    use lichii_common::settings::Settings;
    use lichii_loader::npdm::ServiceAccess;

    use crate::testing::{SWITCH1_TITLE, context, context_with};

    #[test]
    fn names_reference_services_the_title_may_open_that_are_missing() {
        let (context, _client) = context(SWITCH1_TITLE, Settings::default());
        crate::register_all(&context).unwrap();
        let missing = super::missing(&context);
        assert!(missing.contains(&"pm:shell"));
        assert!(!missing.contains(&"fsp-srv"));

        let mut program = lichii_kernel::testing::program_image(SWITCH1_TITLE);
        program.services = ServiceAccess {
            client: vec!["fsp-srv".into(), "hid".into(), "pm:*".into()],
            server: Vec::new(),
        };
        let (context, _client) = context_with(program, Settings::default());
        crate::register_all(&context).unwrap();
        assert_eq!(
            super::missing(&context),
            ["pm:bm", "pm:dmnt", "pm:info", "pm:shell"]
        );
    }
}
