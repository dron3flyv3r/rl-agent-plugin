using System;
using Godot;

namespace RlAgentPlugin.Editor;

internal static class EditorUiScale
{
    private const float MinScale = 0.5f;

    public static float Factor
    {
        get
        {
            try
            {
                return Math.Max(MinScale, EditorInterface.Singleton.GetEditorScale());
            }
            catch
            {
                return 1f;
            }
        }
    }

    public static float Px(float value) => value * Factor;

    public static int Px(int value) => Mathf.RoundToInt(value * Factor);

    public static Vector2 Size(float x, float y) => new(Px(x), Px(y));
}
