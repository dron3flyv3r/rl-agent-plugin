#include "register_types.h"
#include "RlCnnEncoder.h"
#include "RlDenseLayer.h"
#include "RlGruLayer.h"
#include "RlLayerNormLayer.h"
#include "RlLstmLayer.h"
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

using namespace godot;

void initialize_rl_cnn_module(ModuleInitializationLevel p_level)
{
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE)
        return;
    ClassDB::register_class<RlCnnEncoder>();
    ClassDB::register_class<RlDenseLayer>();
    ClassDB::register_class<RlGruLayer>();
    ClassDB::register_class<RlLayerNormLayer>();
    ClassDB::register_class<RlLstmLayer>();
}

void uninitialize_rl_cnn_module(ModuleInitializationLevel p_level)
{
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE)
        return;
}

extern "C" {
GDExtensionBool GDE_EXPORT rl_cnn_library_init(
    GDExtensionInterfaceGetProcAddress p_get_proc_address,
    GDExtensionClassLibraryPtr        p_library,
    GDExtensionInitialization*        r_initialization)
{
    GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);
    init_obj.register_initializer(initialize_rl_cnn_module);
    init_obj.register_terminator(uninitialize_rl_cnn_module);
    init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);
    return init_obj.init();
}
} // extern "C"
