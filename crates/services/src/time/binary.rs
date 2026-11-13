//! The TimeZoneBinary system archive (`0x010000000000080E`): the location list, the rule version
//! and one TZif rule per location, as glue's `TimeZoneBinary` reads them.

use lichii_common::settings::TIME_ZONE_AUTO;
use lichii_fs::romfs::open_romfs;
use lichii_fs::{DirRef, DirectoryExt, FileExt};

use super::zone::{self, Rule};
use crate::ServiceContext;

const TIME_ZONE_BINARY: u64 = 0x0100_0000_0000_080E;
const UTC: &str = "UTC";
/// The location glue checks to tell a readable archive from one decrypted with wrong keys.
const PROBE: &str = "Etc/GMT";

/// `nn::time::LocationName`.
pub(super) type LocationName = [u8; 0x24];
/// `nn::time::RuleVersion`.
pub(super) type RuleVersion = [u8; 0x10];

/// `name` as a location name, if it fits with its terminator.
pub(super) fn location_name(name: &str) -> Option<LocationName> {
    let bytes = name.as_bytes();
    if bytes.is_empty() || bytes.len() >= 0x24 || bytes.contains(&0) {
        return None;
    }
    let mut location = [0; 0x24];
    location[..bytes.len()].copy_from_slice(bytes);
    Some(location)
}

fn name_str(location: &LocationName) -> &str {
    let end = location
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(location.len());
    std::str::from_utf8(&location[..end]).unwrap_or_default()
}

/// The mounted archive. Without a readable one, only UTC exists, with a built-in rule.
pub(super) struct TimeZoneBinary {
    root: Option<DirRef>,
    locations: Vec<LocationName>,
    version: RuleVersion,
}

impl TimeZoneBinary {
    /// Mounts the archive from the NAND's registered content, else the synthesized one.
    pub(super) fn open(context: &ServiceContext) -> Self {
        let root = crate::fs::system_data(context, TIME_ZONE_BINARY).and_then(|file| {
            open_romfs(&file)
                .inspect_err(|e| tracing::error!("opening the time zone archive: {e}"))
                .ok()
        });
        match root {
            Some(root) => Self::from_root(root),
            None => {
                tracing::warn!("no time zone archive; only UTC is available");
                Self::utc_only()
            }
        }
    }

    /// The archive mounted at `root`, or UTC only when it holds no rule for [`PROBE`].
    fn from_root(root: DirRef) -> Self {
        let read = |path: &str| root.open_file_path(path).ok().flatten()?.read_all().ok();
        if read(&format!("zoneinfo/{PROBE}")).is_none_or(|rule| rule.is_empty()) {
            tracing::warn!("the time zone archive has no {PROBE} rule; only UTC is available");
            return Self::utc_only();
        }
        let locations = parse_list(&read("binaryList.txt").unwrap_or_default());
        let mut version = [0; 0x10];
        let text = read("version.txt").unwrap_or_default();
        let len = text.len().min(version.len() - 1);
        version[..len].copy_from_slice(&text[..len]);
        Self {
            root: Some(root),
            locations,
            version,
        }
    }

    fn utc_only() -> Self {
        Self {
            root: None,
            locations: vec![location_name(UTC).unwrap()],
            version: [0; 0x10],
        }
    }

    /// The TZif file of `location`, if the archive has a non-empty one.
    pub(super) fn rule_file(&self, location: &LocationName) -> Option<Vec<u8>> {
        let name = name_str(location);
        let Some(root) = &self.root else {
            return (name == UTC).then(|| zone::utc_tzif().to_vec());
        };
        if name.is_empty() {
            return None;
        }
        root.open_file_path(&format!("zoneinfo/{name}"))
            .ok()
            .flatten()?
            .read_all()
            .ok()
            .filter(|data| !data.is_empty())
    }

    /// `binaryList.txt`: every location, in order.
    pub(super) fn locations(&self) -> &[LocationName] {
        &self.locations
    }

    /// `version.txt`, NUL-terminated.
    pub(super) fn version(&self) -> RuleVersion {
        self.version
    }

    /// The device location for `setting` (`settings.system.time_zone`) and its rule: the named
    /// zone, else the host's zone (`host`, asked only when needed), else UTC.
    pub(super) fn device_location(
        &self,
        setting: &str,
        host: impl FnOnce() -> Option<String>,
    ) -> (LocationName, Rule) {
        let parsed = |name: &str| {
            let location = location_name(name)?;
            let rule = Rule::parse_tzif(&self.rule_file(&location)?)?;
            Some((location, rule))
        };
        if setting != TIME_ZONE_AUTO {
            if let Some(found) = parsed(setting) {
                return found;
            }
            tracing::warn!(
                "time zone {setting:?} is not in the time zone archive; using the host's"
            );
        }
        let host = host();
        if let Some(found) = host.as_deref().and_then(parsed) {
            return found;
        }
        tracing::warn!("host time zone {host:?} is not in the time zone archive; using UTC");
        parsed(UTC).unwrap_or_else(|| {
            let rule = Rule::parse_tzif(&zone::utc_tzif()).expect("the UTC rule parses");
            (location_name(UTC).unwrap(), rule)
        })
    }
}

/// The host's IANA time zone.
pub(super) fn host_zone() -> Option<String> {
    iana_time_zone::get_timezone()
        .inspect_err(|e| tracing::warn!("reading the host time zone: {e}"))
        .ok()
}

/// One location per `\n`-terminated line, as glue parses the list: `\r` is skipped, a NUL ends
/// the list and a name too long for a location is dropped.
fn parse_list(list: &[u8]) -> Vec<LocationName> {
    let end = list.iter().position(|&b| b == 0).unwrap_or(list.len());
    let mut lines: Vec<&[u8]> = list[..end].split(|&b| b == b'\n').collect();
    lines.pop();
    lines
        .into_iter()
        .filter_map(|line| {
            let name: Vec<u8> = line.iter().copied().filter(|&b| b != b'\r').collect();
            let location = std::str::from_utf8(&name).ok().and_then(location_name);
            if location.is_none() {
                tracing::warn!(
                    "skipping time zone list entry {:?}",
                    String::from_utf8_lossy(&name)
                );
            }
            location
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use lichii_fs::MemDir;

    use super::*;

    fn name(location: &LocationName) -> &str {
        name_str(location)
    }

    fn archive(zones: &[(&str, Vec<u8>)], list: &[u8]) -> TimeZoneBinary {
        let mut zoneinfo = MemDir::new();
        let mut etc = MemDir::new();
        for (zone, rule) in zones {
            match zone.strip_prefix("Etc/") {
                Some(zone) => etc = etc.with_bytes(zone, rule.clone()),
                None => zoneinfo = zoneinfo.with_bytes(*zone, rule.clone()),
            }
        }
        let root = MemDir::new()
            .with_bytes("binaryList.txt", list.to_vec())
            .with_bytes("version.txt", b"2025b\n".to_vec())
            .with_dir("zoneinfo", zoneinfo.with_dir("Etc", etc));
        TimeZoneBinary::from_root(root.into_ref())
    }

    fn utc_zones() -> TimeZoneBinary {
        let utc = zone::utc_tzif().to_vec();
        archive(
            &[("Etc/GMT", utc.clone()), ("GMT", utc.clone()), ("UTC", utc)],
            b"Etc/GMT\r\nGMT\nUTC\nunterminated",
        )
    }

    #[test]
    fn the_list_and_version_come_from_the_archive() {
        let binary = utc_zones();
        let names: Vec<&str> = binary.locations().iter().map(name).collect();
        assert_eq!(names, ["Etc/GMT", "GMT", "UTC"]);
        assert_eq!(&binary.version()[..7], b"2025b\n\0");
        assert!(binary.rule_file(&location_name("GMT").unwrap()).is_some());
        assert!(
            binary
                .rule_file(&location_name("Asia/Tokyo").unwrap())
                .is_none()
        );
        assert!(binary.rule_file(&[0; 0x24]).is_none());
    }

    #[test]
    fn the_setting_then_the_host_then_utc() {
        let binary = utc_zones();
        let never = || -> Option<String> { panic!("the host zone is not needed") };
        assert_eq!(name(&binary.device_location("GMT", never).0), "GMT");
        assert_eq!(
            name(
                &binary
                    .device_location("Mars/Olympus", || Some("GMT".into()))
                    .0
            ),
            "GMT"
        );
        assert_eq!(
            name(
                &binary
                    .device_location(TIME_ZONE_AUTO, || Some("Etc/GMT".into()))
                    .0
            ),
            "Etc/GMT"
        );
        assert_eq!(
            name(&binary.device_location("Mars/Olympus", || None).0),
            "UTC"
        );
        assert_eq!(
            name(
                &binary
                    .device_location(TIME_ZONE_AUTO, || Some("Europe/Paris".into()))
                    .0
            ),
            "UTC"
        );
    }

    #[test]
    fn an_archive_without_etc_gmt_holds_utc_only() {
        let binary = archive(&[("UTC", zone::utc_tzif().to_vec())], b"UTC\nGMT\n");
        let names: Vec<&str> = binary.locations().iter().map(name).collect();
        assert_eq!(names, ["UTC"]);
        assert_eq!(binary.version(), [0; 0x10]);
        assert_eq!(name(&binary.device_location("GMT", || None).0), "UTC");
    }
}
