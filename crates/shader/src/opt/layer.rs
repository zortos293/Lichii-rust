//! Emulates layer output from stages before geometry, ported from `layer_pass.cpp`.

use crate::ShaderStage;
use crate::info::ShaderInfo;
use crate::ir::{Attribute, InstId, Opcode, Program, Value};
use crate::profile::HostTranslateInfo;

/// On hosts that cannot write the layer before the geometry stage, redirects layer stores of
/// vertex and tessellation programs to the first free generic output, which a host geometry
/// stage then forwards. Records the chosen attribute in `emulated_layer`.
pub fn layer(
    prog: &mut Program,
    stage: ShaderStage,
    info: &mut ShaderInfo,
    host_info: &HostTranslateInfo,
) {
    let permitted = matches!(
        stage,
        ShaderStage::VertexA
            | ShaderStage::VertexB
            | ShaderStage::TessellationControl
            | ShaderStage::TessellationEval
    );
    if host_info.support_viewport_index_layer || !permitted {
        return;
    }
    let layer_attribute = (0..Attribute::NUM_GENERICS)
        .find(|&index| !info.stores.generic(index))
        .map_or_else(
            || {
                crate::fallback::note("layer", "emulation_no_free_generic");
                Attribute::LAYER
            },
            |index| Attribute::generic(index, 0),
        );
    let layer_stores: Vec<InstId> = prog
        .post_order
        .iter()
        .flat_map(|&block| prog.insts_of(block))
        .filter(|&id| {
            prog.op(id) == Opcode::SetAttribute
                && prog.arg(id, 0) == Value::Attribute(Attribute::LAYER)
        })
        .collect();
    if layer_stores.is_empty() {
        return;
    }
    for id in layer_stores {
        prog.set_arg(id, 0, Value::Attribute(layer_attribute));
    }
    info.requires_layer_emulation = true;
    info.emulated_layer = Some(layer_attribute);
    info.stores.set(Attribute::LAYER, false);
    info.stores.set(layer_attribute, true);
}
