# Stand-in shared fonts

`pl:u` serves these when neither the NAND system archives nor the fonts directory has any shared
font. They are the open-source fonts the C++ reference synthesizes for the font system archives
(`src/core/file_sys/system_archive/data`), byte for byte.

| File | Font types | Font | License |
|---|---|---|---|
| standard.ttf | 0 (standard) | Open Sans Regular 1.10, (c) 2010-2011 Google Corporation | Apache 2.0 (LICENSE-Apache-2.0.txt) |
| chinese_simplified_korean.ttf | 1 (Simplified Chinese), 4 (Korean) | Open Sans Regular 1.10, (c) 2010-2011 Google Corporation | Apache 2.0 (LICENSE-Apache-2.0.txt) |
| chinese_traditional.ttf | 3 (Traditional Chinese) | Open Sans Light 1.10, (c) 2010-2011 Google Corporation | Apache 2.0 (LICENSE-Apache-2.0.txt) |
| extended_chinese_simplified.ttf | 2 (Simplified Chinese extension) | Source Sans Pro Regular 2.021, (c) 2010, 2012, 2014 Adobe Systems Incorporated | SIL OFL 1.1 (OFL.txt) |
| nintendo_extended.ttf | 5, 6 (button icons) | YuzuOSSExtension 1.000, yuzu Emulator Project | GPL-2.0-or-later |

They cover Latin text and the controller button icons, not CJK glyphs.
