//! Ampere operations, one variant per instruction whose operands are laid out differently.

macro_rules! opcodes {
    ($($(#[doc = $doc:literal])* $variant:ident $name:literal $uniform:literal,)*) => {
        /// An Ampere (SM86) operation.
        ///
        /// Aliases that share an encoding (IMAD.MOV, IMAD.SHL, BMOV.CLEAR, ...) are one
        /// variant; forms with a different operand layout (IMAD.WIDE, ATOMS.CAS, SULD.D, the
        /// 64-bit conversions) are their own.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Opcode {
            $($(#[doc = $doc])* $variant,)*
        }

        impl Opcode {
            /// Every operation the decoder knows.
            pub const ALL: &[Opcode] = &[$(Opcode::$variant,)*];

            /// The mnemonic nvdisasm prints, without modifiers.
            pub const fn name(self) -> &'static str {
                match self {
                    $(Opcode::$variant => $name,)*
                }
            }

            /// Whether the operation runs on the uniform datapath: its guard and predicate
            /// operands name uniform predicates and its register operands uniform registers.
            pub const fn is_uniform(self) -> bool {
                match self {
                    $(Opcode::$variant => $uniform,)*
                }
            }
        }
    };
}

opcodes! {
    Fadd "FADD" false,
    Ffma "FFMA" false,
    Fmul "FMUL" false,
    Fmnmx "FMNMX" false,
    Fset "FSET" false,
    Fsetp "FSETP" false,
    Fsel "FSEL" false,
    Fswzadd "FSWZADD" false,
    Mufu "MUFU" false,
    Hadd2 "HADD2" false,
    Hfma2 "HFMA2" false,
    Hmul2 "HMUL2" false,
    Hset2 "HSET2" false,
    Hsetp2 "HSETP2" false,
    Hmnmx2 "HMNMX2" false,
    Dadd "DADD" false,
    Dfma "DFMA" false,
    Dmul "DMUL" false,
    Dsetp "DSETP" false,
    Iadd3 "IADD3" false,
    Imad "IMAD" false,
    /// IMAD.WIDE: a 64-bit result in a register pair.
    ImadWide "IMAD.WIDE" false,
    /// IMAD.HI: the high word of the 64-bit product plus a 64-bit addend.
    ImadHi "IMAD.HI" false,
    Idp "IDP" false,
    Iabs "IABS" false,
    Imnmx "IMNMX" false,
    Isetp "ISETP" false,
    Lop3 "LOP3" false,
    Shf "SHF" false,
    Prmt "PRMT" false,
    Popc "POPC" false,
    Flo "FLO" false,
    Brev "BREV" false,
    Lea "LEA" false,
    Sel "SEL" false,
    Sgxt "SGXT" false,
    Bmsk "BMSK" false,
    Mov "MOV" false,
    P2r "P2R" false,
    R2p "R2P" false,
    Plop3 "PLOP3" false,
    Cs2r "CS2R" false,
    S2r "S2R" false,
    /// Coarse shading mode test; titles test it before an early EXIT.
    Csmtest "CSMTEST" false,
    /// Loads the program counter; geometry programs start with it to install their exit
    /// handler.
    Lepc "LEPC" false,
    F2f "F2F" false,
    /// F2F with a 64-bit source or destination.
    F2f64 "F2F" false,
    F2i "F2I" false,
    /// F2I with a 64-bit source or destination.
    F2i64 "F2I" false,
    I2f "I2F" false,
    /// I2F with a 64-bit source or destination.
    I2f64 "I2F" false,
    Frnd "FRND" false,
    /// FRND of a 64-bit value.
    Frnd64 "FRND" false,
    F2fp "F2FP" false,
    F2ip "F2IP" false,
    I2fp "I2FP" false,
    Umov "UMOV" true,
    Uldc "ULDC" true,
    /// Load from global memory into a uniform register.
    Ldcg "LDCG" true,
    Uiadd3 "UIADD3" true,
    Uimad "UIMAD" true,
    UimadWide "UIMAD.WIDE" true,
    Ulea "ULEA" true,
    Ulop3 "ULOP3" true,
    Ushf "USHF" true,
    Usel "USEL" true,
    Uprmt "UPRMT" true,
    Usgxt "USGXT" true,
    Ubmsk "UBMSK" true,
    Uplop3 "UPLOP3" true,
    Uisetp "UISETP" true,
    Uflo "UFLO" true,
    Upopc "UPOPC" true,
    Ubrev "UBREV" true,
    Up2ur "UP2UR" true,
    Uf2fp "UF2FP" true,
    S2ur "S2UR" true,
    Vote "VOTE" false,
    /// VOTE into a uniform register and predicate, evaluated per thread.
    Voteu "VOTEU" false,
    Shfl "SHFL" false,
    Match "MATCH" false,
    /// Warp reduction of a register into a uniform register.
    Redux "REDUX" false,
    /// Copies a register of the first active thread into a uniform register.
    R2ur "R2UR" false,
    /// Generic address space load.
    Ld "LD" false,
    Ldg "LDG" false,
    Ldl "LDL" false,
    Lds "LDS" false,
    Ldc "LDC" false,
    /// Generic address space store.
    St "ST" false,
    Stg "STG" false,
    Stl "STL" false,
    Sts "STS" false,
    Ldsm "LDSM" false,
    Ldgsts "LDGSTS" false,
    Ldgdepbar "LDGDEPBAR" false,
    Ldtram "LDTRAM" false,
    /// Generic address space atomic.
    Atom "ATOM" false,
    AtomCas "ATOM.CAS" false,
    Atomg "ATOMG" false,
    AtomgCas "ATOMG.CAS" false,
    Atoms "ATOMS" false,
    AtomsCas "ATOMS.CAS" false,
    /// ATOMS.POPC.INC: adds the number of active threads, returning nothing.
    AtomsPopcInc "ATOMS.POPC.INC" false,
    Red "RED" false,
    Cctl "CCTL" false,
    Membar "MEMBAR" false,
    Errbar "ERRBAR" false,
    Depbar "DEPBAR" false,
    Tex "TEX" false,
    Tld "TLD" false,
    Tld4 "TLD4" false,
    Tmml "TMML" false,
    Txd "TXD" false,
    Txq "TXQ" false,
    /// Formatted surface load (SULD.P).
    SuldP "SULD.P" false,
    /// Raw surface load (SULD.D).
    SuldD "SULD.D" false,
    SustP "SUST.P" false,
    SustD "SUST.D" false,
    Suatom "SUATOM" false,
    SuatomCas "SUATOM.CAS" false,
    Sured "SURED" false,
    Ald "ALD" false,
    Ast "AST" false,
    Ipa "IPA" false,
    Isberd "ISBERD" false,
    Isbewr "ISBEWR" false,
    Out "OUT" false,
    Pixld "PIXLD" false,
    Al2p "AL2P" false,
    Bar "BAR" false,
    Warpsync "WARPSYNC" false,
    Nanosleep "NANOSLEEP" false,
    Yield "YIELD" false,
    Nop "NOP" false,
    Bpt "BPT" false,
    Bra "BRA" false,
    Brx "BRX" false,
    /// BRX through a uniform register.
    Brxu "BRXU" false,
    Jmp "JMP" false,
    Jmx "JMX" false,
    /// JMX through a uniform register.
    Jmxu "JMXU" false,
    CallAbs "CALL.ABS" false,
    /// CALL.REL; bit 86 is .NOINC.
    CallRel "CALL.REL" false,
    /// RET.REL; bit 86 is .NODEC.
    Ret "RET" false,
    Exit "EXIT" false,
    Kill "KILL" false,
    Bssy "BSSY" false,
    Bsync "BSYNC" false,
    Break "BREAK" false,
    /// BMOV from a convergence barrier into a register.
    BmovR "BMOV" false,
    /// BMOV from a register into a convergence barrier.
    BmovB "BMOV" false,
    /// The 64-bit BMOV that sets the at-exit handler address: the geometry prologue's
    /// 0x357 and the handler's 0x957 (C++ `geometry_exit.h`).
    BmovAtExit "BMOV" false,
    Hmma "HMMA" false,
    Imma "IMMA" false,
    Movm "MOVM" false,
}
