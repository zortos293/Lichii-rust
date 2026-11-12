//! Maxwell opcodes, in the first-match-wins order of the C++ `maxwell.inc`.

/// The top 16 bits of an instruction an opcode requires: `insn >> 48 & mask == value`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Pattern {
    pub mask: u16,
    pub value: u16,
}

impl Pattern {
    /// Parses a `maxwell.inc` encoding string: `0` and `1` are fixed bits from bit 63 down,
    /// `-` is a don't-care bit, and anything else is ignored.
    const fn parse(encoding: &str) -> Pattern {
        let bytes = encoding.as_bytes();
        let mut mask = 0u16;
        let mut value = 0u16;
        let mut bits = 0;
        let mut i = 0;
        while i < bytes.len() {
            let bit = match bytes[i] {
                b'0' => Some(false),
                b'1' => Some(true),
                b'-' => None,
                _ => {
                    i += 1;
                    continue;
                }
            };
            assert!(bits < 16, "encoding longer than 16 bits");
            let place = 1 << (15 - bits);
            if let Some(set) = bit {
                mask |= place;
                if set {
                    value |= place;
                }
            }
            bits += 1;
            i += 1;
        }
        assert!(bits == 16, "encoding shorter than 16 bits");
        Pattern { mask, value }
    }

    pub const fn matches(self, insn: u64) -> bool {
        (insn >> 48) as u16 & self.mask == self.value
    }
}

macro_rules! opcodes {
    ($($variant:ident, $name:literal, $encoding:literal;)*) => {
        /// A Maxwell instruction, one variant per `INST` line of the C++ `maxwell.inc`.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[repr(u16)]
        pub enum Opcode {
            $($variant,)*
        }

        impl Opcode {
            /// Every opcode in decode priority order.
            pub const ALL: &'static [Opcode] = &[$(Opcode::$variant,)*];
            const NAMES: &'static [&'static str] = &[$($name,)*];
            const PATTERNS: &'static [Pattern] = &[$(Pattern::parse($encoding),)*];
        }
    };
}

impl Opcode {
    /// The C++ display name, such as `"FADD (reg)"`.
    pub const fn name(self) -> &'static str {
        Self::NAMES[self as usize]
    }

    pub const fn pattern(self) -> Pattern {
        Self::PATTERNS[self as usize]
    }
}

impl std::fmt::Display for Opcode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

opcodes! {
    TensorMulAdd, "TENSOR_MUL_ADD", "0101 1101 0011 0---";
    Cctlt,        "CCTLT",          "1110 1011 1111 0--0";
    Al2p,         "AL2P",           "1110 1111 1010 0---";
    Ald,          "ALD",            "1110 1111 1101 1---";
    Ast,          "AST",            "1110 1111 1111 0---";
    B2r,          "B2R",            "1111 0000 1011 1---";
    Bar,          "BAR",            "1111 0000 1010 1---";
    BfeCbuf,      "BFE (cbuf)",     "0100 1100 0000 0---";
    BfeReg,       "BFE (reg)",      "0101 1100 0000 0---";
    BfiCr,        "BFI (cr)",       "0100 1011 1111 0---";
    BfiRc,        "BFI (rc)",       "0101 0011 1111 0---";
    BfiReg,       "BFI (reg)",      "0101 1011 1111 0---";
    Cs2r,         "CS2R",           "0101 0000 1100 1---";
    Cset,         "CSET",           "0101 0000 1001 1---";
    Csetp,        "CSETP",          "0101 0000 1010 0---";
    DaddCbuf,     "DADD (cbuf)",    "0100 1100 0111 0---";
    DaddReg,      "DADD (reg)",     "0101 1100 0111 0---";
    Depbar,       "DEPBAR",         "1111 0000 1111 0---";
    DmnmxCbuf,    "DMNMX (cbuf)",   "0100 1100 0101 0---";
    DmnmxReg,     "DMNMX (reg)",    "0101 1100 0101 0---";
    DmulCbuf,     "DMUL (cbuf)",    "0100 1100 1000 0---";
    DmulReg,      "DMUL (reg)",     "0101 1100 1000 0---";
    F2fCbuf,      "F2F (cbuf)",     "0100 1100 1010 1---";
    F2fReg,       "F2F (reg)",      "0101 1100 1010 1---";
    F2iCbuf,      "F2I (cbuf)",     "0100 1100 1011 0---";
    F2iReg,       "F2I (reg)",      "0101 1100 1011 0---";
    FaddCbuf,     "FADD (cbuf)",    "0100 1100 0101 1---";
    FaddReg,      "FADD (reg)",     "0101 1100 0101 1---";
    FchkCbuf,     "FCHK (cbuf)",    "0100 1100 1000 1---";
    FchkReg,      "FCHK (reg)",     "0101 1100 1000 1---";
    FloCbuf,      "FLO (cbuf)",     "0100 1100 0011 0---";
    FloReg,       "FLO (reg)",      "0101 1100 0011 0---";
    FmnmxCbuf,    "FMNMX (cbuf)",   "0100 1100 0110 0---";
    FmnmxReg,     "FMNMX (reg)",    "0101 1100 0110 0---";
    FmulCbuf,     "FMUL (cbuf)",    "0100 1100 0110 1---";
    FmulReg,      "FMUL (reg)",     "0101 1100 0110 1---";
    Fswzadd,      "FSWZADD",        "0101 0000 1111 1---";
    Hadd2Reg,     "HADD2 (reg)",    "0101 1101 0001 0---";
    Hfma2Reg,     "HFMA2 (reg)",    "0101 1101 0000 0---";
    Hmul2Reg,     "HMUL2 (reg)",    "0101 1101 0000 1---";
    Hset2Reg,     "HSET2 (reg)",    "0101 1101 0001 1---";
    Hsetp2Reg,    "HSETP2 (reg)",   "0101 1101 0010 0---";
    I2fCbuf,      "I2F (cbuf)",     "0100 1100 1011 1---";
    I2fReg,       "I2F (reg)",      "0101 1100 1011 1---";
    I2iCbuf,      "I2I (cbuf)",     "0100 1100 1110 0---";
    I2iReg,       "I2I (reg)",      "0101 1100 1110 0---";
    IaddCbuf,     "IADD (cbuf)",    "0100 1100 0001 0---";
    IaddReg,      "IADD (reg)",     "0101 1100 0001 0---";
    IdpImm,       "IDP (imm)",      "0101 0011 1101 1---";
    IdpReg,       "IDP (reg)",      "0101 0011 1111 1---";
    ImnmxCbuf,    "IMNMX (cbuf)",   "0100 1100 0010 0---";
    ImnmxReg,     "IMNMX (reg)",    "0101 1100 0010 0---";
    ImulCbuf,     "IMUL (cbuf)",    "0100 1100 0011 1---";
    ImulReg,      "IMUL (reg)",     "0101 1100 0011 1---";
    Isberd,       "ISBERD",         "1110 1111 1101 0---";
    IsbeNx2,      "ISBE (nx2)",     "1110 1111 1100 0---";
    IscaddCbuf,   "ISCADD (cbuf)",  "0100 1100 0001 1---";
    IscaddReg,    "ISCADD (reg)",   "0101 1100 0001 1---";
    Ldc,          "LDC",            "1110 1111 1001 0---";
    Ldg,          "LDG",            "1110 1110 1101 0---";
    Ldl,          "LDL",            "1110 1111 0100 0---";
    Lds,          "LDS",            "1110 1111 0100 1---";
    LeaHiReg,     "LEA (hi reg)",   "0101 1011 1101 1---";
    LeaLoReg,     "LEA (lo reg)",   "0101 1011 1101 0---";
    Lepc,         "LEPC",           "0101 0000 1101 0---";
    Lop3Reg,      "LOP3 (reg)",     "0101 1011 1110 0---";
    LopCbuf,      "LOP (cbuf)",     "0100 1100 0100 0---";
    LopReg,       "LOP (reg)",      "0101 1100 0100 0---";
    Membar,       "MEMBAR",         "1110 1111 1001 1---";
    MovCbuf,      "MOV (cbuf)",     "0100 1100 1001 1---";
    MovReg,       "MOV (reg)",      "0101 1100 1001 1---";
    Mufu,         "MUFU",           "0101 0000 1000 0---";
    Nop,          "NOP",            "0101 0000 1011 0---";
    OutCbuf,      "OUT (cbuf)",     "1110 1011 1110 0---";
    OutReg,       "OUT (reg)",      "1111 1011 1110 0---";
    P2rCbuf,      "P2R (cbuf)",     "0100 1100 1110 1---";
    P2rImm,       "P2R (imm)",      "0011 1000 1110 1---";
    P2rReg,       "P2R (reg)",      "0101 1100 1110 1---";
    Pixld,        "PIXLD",          "1110 1111 1110 1---";
    PopcCbuf,     "POPC (cbuf)",    "0100 1100 0000 1---";
    PopcReg,      "POPC (reg)",     "0101 1100 0000 1---";
    Pset,         "PSET",           "0101 0000 1000 1---";
    Psetp,        "PSETP",          "0101 0000 1001 0---";
    R2b,          "R2B",            "1111 0000 1100 0---";
    R2pCbuf,      "R2P (cbuf)",     "0100 1100 1111 0---";
    R2pReg,       "R2P (reg)",      "0101 1100 1111 0---";
    Red,          "RED",            "1110 1011 1111 1---";
    RroCbuf,      "RRO (cbuf)",     "0100 1100 1001 0---";
    RroReg,       "RRO (reg)",      "0101 1100 1001 0---";
    S2r,          "S2R",            "1111 0000 1100 1---";
    SelCbuf,      "SEL (cbuf)",     "0100 1100 1010 0---";
    SelReg,       "SEL (reg)",      "0101 1100 1010 0---";
    Shfl,         "SHFL",           "1110 1111 0001 0---";
    ShfLReg,      "SHF (l reg)",    "0101 1011 1111 1---";
    ShfRReg,      "SHF (r reg)",    "0101 1100 1111 1---";
    ShlCbuf,      "SHL (cbuf)",     "0100 1100 0100 1---";
    ShlReg,       "SHL (reg)",      "0101 1100 0100 1---";
    ShrCbuf,      "SHR (cbuf)",     "0100 1100 0010 1---";
    ShrReg,       "SHR (reg)",      "0101 1100 0010 1---";
    Stg,          "STG",            "1110 1110 1101 1---";
    Stl,          "STL",            "1110 1111 0101 0---";
    Stp,          "STP",            "1110 1110 1010 0---";
    Sts,          "STS",            "1110 1111 0101 1---";
    Sync,         "SYNC",           "1111 0000 1111 1---";
    Tmml,         "TMML",           "1101 1111 0101 1---";
    TmmlB,        "TMML (b)",       "1101 1111 0110 0---";
    Txa,          "TXA",            "1101 1111 0100 0---";
    Txq,          "TXQ",            "1101 1111 0100 1---";
    TxqB,         "TXQ (b)",        "1101 1111 0101 0---";
    Vote,         "VOTE",           "0101 0000 1101 1---";
    VoteVtg,      "VOTE (vtg)",     "0101 0000 1110 0---";
    Vsetp,        "VSETP",          "0101 0000 1111 0---";
    AtomCas,      "ATOM (cas)",     "1110 1110 1111 ----";
    BfeImm,       "BFE (imm)",      "0011 100- 0000 0---";
    BfiImm,       "BFI (imm)",      "0011 011- 1111 0---";
    Bpt,          "BPT",            "1110 0011 1010 ----";
    Bra,          "BRA",            "1110 0010 0100 ----";
    Brk,          "BRK",            "1110 0011 0100 ----";
    Brx,          "BRX",            "1110 0010 0101 ----";
    Cal,          "CAL",            "1110 0010 0110 ----";
    Cont,         "CONT",           "1110 0011 0101 ----";
    DaddImm,      "DADD (imm)",     "0011 100- 0111 0---";
    DfmaCr,       "DFMA (cr)",      "0100 1011 0111 ----";
    DfmaRc,       "DFMA (rc)",      "0101 0011 0111 ----";
    DfmaReg,      "DFMA (reg)",     "0101 1011 0111 ----";
    DmnmxImm,     "DMNMX (imm)",    "0011 100- 0101 0---";
    DmulImm,      "DMUL (imm)",     "0011 100- 1000 0---";
    DsetpCbuf,    "DSETP (cbuf)",   "0100 1011 1000 ----";
    DsetpReg,     "DSETP (reg)",    "0101 1011 1000 ----";
    Exit,         "EXIT",           "1110 0011 0000 ----";
    F2fImm,       "F2F (imm)",      "0011 100- 1010 1---";
    F2iImm,       "F2I (imm)",      "0011 100- 1011 0---";
    FaddImm,      "FADD (imm)",     "0011 100- 0101 1---";
    FchkImm,      "FCHK (imm)",     "0011 100- 1000 1---";
    FcmpCr,       "FCMP (cr)",      "0100 1011 1010 ----";
    FcmpRc,       "FCMP (rc)",      "0101 0011 1010 ----";
    FcmpReg,      "FCMP (reg)",     "0101 1011 1010 ----";
    FloImm,       "FLO (imm)",      "0011 100- 0011 0---";
    FmnmxImm,     "FMNMX (imm)",    "0011 100- 0110 0---";
    FmulImm,      "FMUL (imm)",     "0011 100- 0110 1---";
    FsetpCbuf,    "FSETP (cbuf)",   "0100 1011 1011 ----";
    FsetpReg,     "FSETP (reg)",    "0101 1011 1011 ----";
    Getcrsptr,    "GETCRSPTR",      "1110 0010 1100 ----";
    Getlmembase,  "GETLMEMBASE",    "1110 0010 1101 ----";
    I2fImm,       "I2F (imm)",      "0011 100- 1011 1---";
    I2iImm,       "I2I (imm)",      "0011 100- 1110 0---";
    Iadd3Cbuf,    "IADD3 (cbuf)",   "0100 1100 1100 ----";
    Iadd3Reg,     "IADD3 (reg)",    "0101 1100 1100 ----";
    IaddImm,      "IADD (imm)",     "0011 100- 0001 0---";
    IcmpCr,       "ICMP (cr)",      "0100 1011 0100 ----";
    IcmpRc,       "ICMP (rc)",      "0101 0011 0100 ----";
    IcmpReg,      "ICMP (reg)",     "0101 1011 0100 ----";
    Ide,          "IDE",            "1110 0011 1001 ----";
    ImnmxImm,     "IMNMX (imm)",    "0011 100- 0010 0---";
    ImulImm,      "IMUL (imm)",     "0011 100- 0011 1---";
    IscaddImm,    "ISCADD (imm)",   "0011 100- 0001 1---";
    IsetpCbuf,    "ISETP (cbuf)",   "0100 1011 0110 ----";
    IsetpReg,     "ISETP (reg)",    "0101 1011 0110 ----";
    IsetCbuf,     "ISET (cbuf)",    "0100 1011 0101 ----";
    IsetReg,      "ISET (reg)",     "0101 1011 0101 ----";
    Jcal,         "JCAL",           "1110 0010 0010 ----";
    Jmp,          "JMP",            "1110 0010 0001 ----";
    Jmx,          "JMX",            "1110 0010 0000 ----";
    Kil,          "KIL",            "1110 0011 0011 ----";
    LeaLoCbuf,    "LEA (lo cbuf)",  "0100 1011 1101 ----";
    LeaLoImm,     "LEA (lo imm)",   "0011 011- 1101 0---";
    Longjmp,      "LONGJMP",        "1110 0011 0001 ----";
    LopImm,       "LOP (imm)",      "0011 100- 0100 0---";
    Mov32i,       "MOV32I",         "0000 0001 0000 ----";
    MovImm,       "MOV (imm)",      "0011 100- 1001 1---";
    OutImm,       "OUT (imm)",      "1111 011- 1110 0---";
    Pbk,          "PBK",            "1110 0010 1010 ----";
    Pcnt,         "PCNT",           "1110 0010 1011 ----";
    Pexit,        "PEXIT",          "1110 0010 0011 ----";
    Plongjmp,     "PLONGJMP",       "1110 0010 1000 ----";
    PopcImm,      "POPC (imm)",     "0011 100- 0000 1---";
    Pret,         "PRET",           "1110 0010 0111 ----";
    PrmtCr,       "PRMT (cr)",      "0100 1011 1100 ----";
    PrmtRc,       "PRMT (rc)",      "0101 0011 1100 ----";
    PrmtReg,      "PRMT (reg)",     "0101 1011 1100 ----";
    R2pImm,       "R2P (imm)",      "0011 100- 1111 0---";
    Ram,          "RAM",            "1110 0011 1000 ----";
    Ret,          "RET",            "1110 0011 0010 ----";
    RroImm,       "RRO (imm)",      "0011 100- 1001 0---";
    Rtt,          "RTT",            "1110 0011 0110 ----";
    Sam,          "SAM",            "1110 0011 0111 ----";
    SelImm,       "SEL (imm)",      "0011 100- 1010 0---";
    Setcrsptr,    "SETCRSPTR",      "1110 0010 1110 ----";
    Setlmembase,  "SETLMEMBASE",    "1110 0010 1111 ----";
    ShfLImm,      "SHF (l imm)",    "0011 011- 1111 1---";
    ShfRImm,      "SHF (r imm)",    "0011 100- 1111 1---";
    ShlImm,       "SHL (imm)",      "0011 100- 0100 1---";
    ShrImm,       "SHR (imm)",      "0011 100- 0010 1---";
    Ssy,          "SSY",            "1110 0010 1001 ----";
    Cctl,         "CCTL",           "1110 1111 011- ----";
    Cctll,        "CCTLL",          "1110 1111 100- ----";
    DfmaImm,      "DFMA (imm)",     "0011 011- 0111 ----";
    DsetpImm,     "DSETP (imm)",    "0011 011- 1000 ----";
    FcmpImm,      "FCMP (imm)",     "0011 011- 1010 ----";
    FsetpImm,     "FSETP (imm)",    "0011 011- 1011 ----";
    Iadd3Imm,     "IADD3 (imm)",    "0011 100- 1100 ----";
    IcmpImm,      "ICMP (imm)",     "0011 011- 0100 ----";
    IsetpImm,     "ISETP (imm)",    "0011 011- 0110 ----";
    IsetImm,      "ISET (imm)",     "0011 011- 0101 ----";
    PrmtImm,      "PRMT (imm)",     "0011 011- 1100 ----";
    Suld,         "SULD",           "1110 1011 000- ----";
    Sured,        "SURED",          "1110 1011 010- ----";
    Sust,         "SUST",           "1110 1011 001- ----";
    TexB,         "TEX (b)",        "1101 1110 10-- ----";
    Tld4B,        "TLD4 (b)",       "1101 1110 11-- ----";
    Txd,          "TXD",            "1101 1110 00-- ----";
    TxdB,         "TXD (b)",        "1101 1110 01-- ----";
    XmadReg,      "XMAD (reg)",     "0101 1011 00-- ----";
    DsetCbuf,     "DSET (cbuf)",    "0100 1001 0--- ----";
    DsetReg,      "DSET (reg)",     "0101 1001 0--- ----";
    FfmaCr,       "FFMA (cr)",      "0100 1001 1--- ----";
    FfmaRc,       "FFMA (rc)",      "0101 0001 1--- ----";
    FfmaReg,      "FFMA (reg)",     "0101 1001 1--- ----";
    ImadspCr,     "IMADSP (cr)",    "0100 1010 1--- ----";
    ImadspRc,     "IMADSP (rc)",    "0101 0010 1--- ----";
    ImadspReg,    "IMADSP (reg)",   "0101 1010 1--- ----";
    ImadCr,       "IMAD (cr)",      "0100 1010 0--- ----";
    ImadRc,       "IMAD (rc)",      "0101 0010 0--- ----";
    ImadReg,      "IMAD (reg)",     "0101 1010 0--- ----";
    Suatom,       "SUATOM",         "1110 1010 0--- ----";
    SuatomCas,    "SUATOM_cas",     "1110 1010 1--- ----";
    Tld4s,        "TLD4S",          "1101 1111 -0-- ----";
    Vabsdiff4,    "VABSDIFF4",      "0101 0000 0--- ----";
    XmadImm,      "XMAD (imm)",     "0011 011- 00-- ----";
    XmadRc,       "XMAD (rc)",      "0101 0001 0--- ----";
    Atom,         "ATOM",           "1110 1101 ---- ----";
    Atoms,        "ATOMS",          "1110 1100 ---- ----";
    AtomsCas,     "ATOMS (cas)",    "1110 1110 ---- ----";
    DsetImm,      "DSET (imm)",     "0011 001- 0--- ----";
    FfmaImm,      "FFMA (imm)",     "0011 001- 1--- ----";
    Fmul32i,      "FMUL32I",        "0001 1110 ---- ----";
    FsetCbuf,     "FSET (cbuf)",    "0100 1000 ---- ----";
    FsetReg,      "FSET (reg)",     "0101 1000 ---- ----";
    Hadd2Cbuf,    "HADD2 (cbuf)",   "0111 101- 1--- ----";
    Hadd2Imm,     "HADD2 (imm)",    "0111 101- 0--- ----";
    Hmul2Cbuf,    "HMUL2 (cbuf)",   "0111 100- 1--- ----";
    Hmul2Imm,     "HMUL2 (imm)",    "0111 100- 0--- ----";
    Hset2Cbuf,    "HSET2 (cbuf)",   "0111 110- 1--- ----";
    Hset2Imm,     "HSET2 (imm)",    "0111 110- 0--- ----";
    Hsetp2Cbuf,   "HSETP2 (cbuf)",  "0111 111- 1--- ----";
    Hsetp2Imm,    "HSETP2 (imm)",   "0111 111- 0--- ----";
    ImadspImm,    "IMADSP (imm)",   "0011 010- 1--- ----";
    ImadImm,      "IMAD (imm)",     "0011 010- 0--- ----";
    Imul32i,      "IMUL32I",        "0001 1111 ---- ----";
    Ipa,          "IPA",            "1110 0000 ---- ----";
    Tld,          "TLD",            "1101 1100 ---- ----";
    TldB,         "TLD (b)",        "1101 1101 ---- ----";
    Vabsdiff,     "VABSDIFF",       "0101 0100 ---- ----";
    Vmad,         "VMAD",           "0101 1111 ---- ----";
    Vshl,         "VSHL",           "0101 0111 ---- ----";
    Vshr,         "VSHR",           "0101 0110 ---- ----";
    FsetImm,      "FSET (imm)",     "0011 000- ---- ----";
    Hadd2_32i,    "HADD2_32I",      "0010 110- ---- ----";
    Hfma2_32i,    "HFMA2_32I",      "0010 100- ---- ----";
    Hmul2_32i,    "HMUL2_32I",      "0010 101- ---- ----";
    Iadd32i,      "IADD32I",        "0001 110- ---- ----";
    Lop3Cbuf,     "LOP3 (cbuf)",    "0000 001- ---- ----";
    Vmnmx,        "VMNMX",          "0011 101- ---- ----";
    Vset,         "VSET",           "0100 000- ---- ----";
    XmadCr,       "XMAD (cr)",      "0100 111- ---- ----";
    Fadd32i,      "FADD32I",        "0000 10-- ---- ----";
    Ffma32i,      "FFMA32I",        "0000 11-- ---- ----";
    Hfma2Cr,      "HFMA2 (cr)",     "0111 0--- 1--- ----";
    Hfma2Imm,     "HFMA2 (imm)",    "0111 0--- 0--- ----";
    Hfma2Rc,      "HFMA2 (rc)",     "0110 0--- 1--- ----";
    Imad32i,      "IMAD32I",        "1000 00-- ---- ----";
    Iscadd32i,    "ISCADD32I",      "0001 01-- ---- ----";
    LeaHiCbuf,    "LEA (hi cbuf)",  "0001 10-- ---- ----";
    Lop32i,       "LOP32I",         "0000 01-- ---- ----";
    Lop3Imm,      "LOP3 (imm)",     "0011 11-- ---- ----";
    Texs,         "TEXS",           "1101 -00- ---- ----";
    Tld4,         "TLD4",           "1100 10-- ---- ----";
    Tlds,         "TLDS",           "1101 -01- ---- ----";
    Vadd,         "VADD",           "0010 00-- ---- ----";
    Tex,          "TEX",            "1100 0--- ---- ----";
    Ld,           "LD",             "100- ---- ---- ----";
    St,           "ST",             "101- ---- ---- ----";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_mirrors_maxwell_inc() {
        assert_eq!(Opcode::ALL.len(), 282);
        assert_eq!(Opcode::NAMES.len(), Opcode::ALL.len());
        assert_eq!(Opcode::PATTERNS.len(), Opcode::ALL.len());
        for (index, &opcode) in Opcode::ALL.iter().enumerate() {
            assert_eq!(opcode as usize, index);
        }
        assert_eq!(Opcode::ALL[0], Opcode::TensorMulAdd);
        assert_eq!(Opcode::ALL[281], Opcode::St);
    }

    #[test]
    fn names() {
        assert_eq!(Opcode::FaddReg.name(), "FADD (reg)");
        assert_eq!(Opcode::Fadd32i.name(), "FADD32I");
        assert_eq!(Opcode::Hadd2_32i.name(), "HADD2_32I");
        assert_eq!(Opcode::SuatomCas.name(), "SUATOM_cas");
        assert_eq!(Opcode::LeaHiCbuf.to_string(), "LEA (hi cbuf)");
    }

    #[test]
    fn patterns() {
        assert_eq!(
            Opcode::FaddReg.pattern(),
            Pattern {
                mask: 0xfff8,
                value: 0x5c58
            }
        );
        assert_eq!(
            Opcode::Cctlt.pattern(),
            Pattern {
                mask: 0xfff9,
                value: 0xebf0
            }
        );
        assert_eq!(
            Opcode::Tld4s.pattern(),
            Pattern {
                mask: 0xff40,
                value: 0xdf00
            }
        );
        assert_eq!(
            Opcode::Ld.pattern(),
            Pattern {
                mask: 0xe000,
                value: 0x8000
            }
        );
        assert!(Opcode::Exit.pattern().matches(0xe300_0000_0007_000f));
        assert!(!Opcode::Exit.pattern().matches(0xe310_0000_0007_000f));
    }
}
