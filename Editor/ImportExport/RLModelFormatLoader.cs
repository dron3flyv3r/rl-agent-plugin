using System;
using Godot;
using RlAgentPlugin.Runtime;

namespace RlAgentPlugin.Editor;

[Tool]
public partial class RLModelFormatLoader : ResourceFormatLoader
{
    private const string Extension = "rlmodel";

    public override string[] _GetRecognizedExtensions() => [Extension];

    public override string _GetResourceType(string path)
    {
        return path.EndsWith($".{Extension}", StringComparison.OrdinalIgnoreCase)
            ? nameof(RLCheckpoint)
            : string.Empty;
    }

    public override bool _HandlesType(StringName type)
    {
        var typeName = type.ToString();
        return typeName == nameof(RLCheckpoint) || typeName == nameof(Resource);
    }

    public override Variant _Load(string path, string originalPath = "", bool useSubThreads = false, int cacheMode = 0)
    {
        var checkpoint = RLModelLoader.LoadFromFile(path);
        if (checkpoint is null)
        {
            return Variant.From((int)Error.FileCantOpen);
        }

        checkpoint.ResourcePath = path;
        return Variant.From(checkpoint);
    }
}
