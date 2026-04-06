using System;
using System.Collections.Generic;
using System.IO;

namespace RlAgentPlugin.Runtime;

public static class RLSetupWizardDefaults
{
    public static string SanitizeIdentifier(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        var builder = new System.Text.StringBuilder(value.Length);
        foreach (var character in value)
        {
            if (char.IsLetterOrDigit(character))
            {
                builder.Append(char.ToLowerInvariant(character));
            }
            else if (character is '-' or '_')
            {
                builder.Append(character);
            }
        }

        return builder.ToString().Trim('_', '-');
    }

    public static string MakeUniqueIdentifier(string value, IEnumerable<string> existing, string fallback = "agent")
    {
        var baseId = SanitizeIdentifier(value);
        if (string.IsNullOrWhiteSpace(baseId))
        {
            baseId = fallback;
        }

        var existingSet = new HashSet<string>(existing ?? Array.Empty<string>(), StringComparer.Ordinal);
        var candidate = baseId;
        var suffix = 2;
        while (existingSet.Contains(candidate))
        {
            candidate = $"{baseId}_{suffix++}";
        }

        return candidate;
    }

    public static string MakeUniqueFileName(string desiredFileName, IEnumerable<string> existingFileNames)
    {
        var extension = Path.GetExtension(desiredFileName);
        var stem = Path.GetFileNameWithoutExtension(desiredFileName);
        var existingSet = new HashSet<string>(existingFileNames ?? Array.Empty<string>(), StringComparer.Ordinal);
        var candidate = desiredFileName;
        var suffix = 2;
        while (existingSet.Contains(candidate))
        {
            candidate = $"{stem}_{suffix++}{extension}";
        }

        return candidate;
    }
}
