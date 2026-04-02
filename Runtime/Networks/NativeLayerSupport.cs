using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Checks whether the native layer GDExtension classes (<c>RlDenseLayer</c>,
/// <c>RlLayerNormLayer</c>) are available at runtime. The check is performed
/// once at first access and cached.
///
/// Availability requires:
///   - The native library is built (cmake --build).
///   - <c>rl_cnn.gdextension</c> is present in the Godot project and loaded.
/// </summary>
internal static class NativeLayerSupport
{
    /// <summary>
    /// True if <c>RlDenseLayer</c> can be instantiated via ClassDB.
    /// False if the native library is absent or not yet loaded.
    /// </summary>
    public static bool IsAvailable { get; } = CheckAvailability();

    private static bool CheckAvailability()
    {
        try
        {
            var obj = ClassDB.Instantiate("RlDenseLayer").AsGodotObject();
            if (obj is null) return false;
            obj.Dispose();
            return true;
        }
        catch
        {
            return false;
        }
    }
}
