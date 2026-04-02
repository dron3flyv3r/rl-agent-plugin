using System;
using System.Collections.Generic;
using System.Reflection;
using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// Writes HPO trial parameters to a JSON override file and applies them to a
/// <see cref="RLTrainerConfig"/> instance at training startup.
/// <para>
/// Property names must match <see cref="RLTrainerConfig"/> exactly (case-sensitive).
/// For example: <c>"LearningRate"</c>, <c>"ClipEpsilon"</c>, <c>"SacTau"</c>.
/// </para>
/// </summary>
public static class HPOConfigApplicator
{
    // ── Write override file ───────────────────────────────────────────────

    /// <summary>
    /// Serialises the trial's parameter map to a JSON file at <paramref name="resPath"/>.
    /// The file is a flat JSON object: <c>{"LearningRate": 0.0003, "Gamma": 0.99, ...}</c>.
    /// </summary>
    public static void WriteOverrideFile(string resPath, Dictionary<string, float> parameters)
    {
        EnsureParentDirectory(resPath);
        using var file = FileAccess.Open(resPath, FileAccess.ModeFlags.Write);
        if (file is null)
        {
            GD.PushError($"[HPO] Could not write override file '{resPath}': {FileAccess.GetOpenError()}");
            return;
        }

        var dict = new Godot.Collections.Dictionary();
        foreach (var (key, value) in parameters)
            dict[key] = value;

        file.StoreString(Json.Stringify(dict));
    }

    // ── Apply override file to runtime config ────────────────────────────

    /// <summary>
    /// Reads the JSON override file produced by <see cref="WriteOverrideFile"/> and
    /// applies the values to <paramref name="config"/> via reflection.
    /// Unknown keys are skipped with a warning.
    /// </summary>
    public static void ApplyOverrides(string resPath, RLTrainerConfig config)
    {
        if (!FileAccess.FileExists(resPath))
        {
            GD.PushWarning($"[HPO] Override file not found at '{resPath}'.");
            return;
        }

        using var file = FileAccess.Open(resPath, FileAccess.ModeFlags.Read);
        if (file is null) return;

        var parsed = Json.ParseString(file.GetAsText());
        if (parsed.VariantType != Variant.Type.Dictionary) return;

        var configType = typeof(RLTrainerConfig);
        foreach (var (keyVar, valVar) in parsed.AsGodotDictionary())
        {
            string key = keyVar.ToString();
            if (valVar.VariantType is not (Variant.Type.Float or Variant.Type.Int))
                continue;

            float rawValue = valVar.VariantType == Variant.Type.Int
                ? (int)valVar
                : (float)(double)valVar;

            var prop = configType.GetProperty(key, BindingFlags.Public | BindingFlags.Instance);
            if (prop is null || !prop.CanWrite)
            {
                GD.PushWarning($"[HPO] Unknown or read-only RLTrainerConfig property '{key}' — skipped.");
                continue;
            }

            try
            {
                object boxed = prop.PropertyType switch
                {
                    Type t when t == typeof(float) => rawValue,
                    Type t when t == typeof(int)   => (int)Math.Round(rawValue),
                    Type t when t == typeof(bool)  => rawValue > 0.5f,
                    Type t when t == typeof(long)  => (long)Math.Round(rawValue),
                    _ => Convert.ChangeType(rawValue, prop.PropertyType),
                };
                prop.SetValue(config, boxed);
            }
            catch (Exception ex)
            {
                GD.PushWarning($"[HPO] Could not set '{key}' to {rawValue}: {ex.Message}");
            }
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private static void EnsureParentDirectory(string resPath)
    {
        var normalized = resPath.Replace('\\', '/');
        int slash = normalized.LastIndexOf('/');
        if (slash < 0) return;
        var dirPart = normalized[..slash];
        DirAccess.MakeDirRecursiveAbsolute(ProjectSettings.GlobalizePath(dirPart));
    }
}
