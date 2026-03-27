using Godot;
using Godot.Collections;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// A camera sensor node for 2D scenes. Place it anywhere in your scene and position/rotate
/// it like a regular Camera2D. Call <c>buffer.AddImage("name", this)</c> from your agent's
/// <c>CollectObservations</c> override to feed the captured pixels into the observation.
///
/// Internally the sensor owns a <see cref="SubViewport"/> that renders the world from a
/// mirrored <see cref="Camera2D"/>. Each <see cref="Capture"/> call returns the most recent
/// frame after applying the <see cref="Transforms"/> pipeline in order.
///
/// <b>Transform pipeline</b><br/>
/// Add <see cref="RLCameraTransform"/> resources to <see cref="Transforms"/> to pre-process
/// the image. Transforms are applied left-to-right. Built-in transforms:
/// <list type="bullet">
///   <item><see cref="RLGrayscaleTransform"/> — convert to single-channel luminance.</item>
///   <item><see cref="RLCropTransform"/> — crop a rectangular region.</item>
///   <item><see cref="RLResizeTransform"/> — scale to a target resolution.</item>
///   <item><see cref="RLFlipTransform"/> — mirror horizontally or vertically.</item>
/// </list>
/// Subclass <see cref="RLCameraTransform"/> and mark it <c>[GlobalClass]</c> to add your own.
///
/// Editor overlay: the white rectangle shows the full render area. If a
/// <see cref="RLCropTransform"/> is present in the pipeline, its crop region is drawn in yellow.
/// </summary>
[Tool]
[GlobalClass]
public partial class RLCameraSensor2D : Node2D
{
    // ── Exports ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Optional CNN/MLP encoder config for this sensor's image stream.
    /// When set, it is bound to the stream automatically when the agent calls
    /// <c>buffer.AddImage(this)</c>.
    /// </summary>
    [Export(PropertyHint.ResourceType, nameof(RLStreamEncoderConfig))]
    public RLStreamEncoderConfig? EncoderConfig { get; set; }

    [Export]
    public Vector2I RenderSize
    {
        get => _renderSize;
        set { _renderSize = value; _dirty = true; if (_viewport != null) _viewport.Size = value; QueueRedraw(); }
    }

    /// <summary>Zoom of the internal camera. 1 = world units match viewport pixels.</summary>
    [Export]
    public float Zoom
    {
        get => _zoom;
        set { _zoom = Mathf.Max(0.001f, value); _dirty = true; QueueRedraw(); }
    }

    /// <summary>
    /// Image transforms applied in order after the viewport is captured.
    /// Each transform receives the output of the previous one.
    /// Use <see cref="RLGrayscaleTransform"/>, <see cref="RLCropTransform"/>,
    /// <see cref="RLResizeTransform"/>, <see cref="RLFlipTransform"/>, or a custom subclass.
    /// </summary>
    [Export(PropertyHint.ResourceType, nameof(RLCameraTransform))]
    public Array<Resource> Transforms
    {
        get => _transforms;
        set { _transforms = value; QueueRedraw(); }
    }

    // ── Private state ────────────────────────────────────────────────────────

    private Vector2I         _renderSize = new(128, 128);
    private float            _zoom       = 1.0f;
    private Array<Resource>  _transforms = new();

    private SubViewport? _viewport;
    private Camera2D?    _camera;

    // Pixel cache — refreshed once per rendered frame in _Process, not per physics step.
    private byte[]? _cachedPixels;
    private ImageTexture? _finalDebugTexture;
    private bool          _debugTextureRequested;

    private bool _dirty = true;

    // Overlay colours.
    private static readonly Color CRenderFill   = new(0.5f, 0.8f, 1f,   0.06f);
    private static readonly Color CRenderBorder = new(1f,   1f,   1f,   0.35f);
    private static readonly Color CCropBorder   = new(1f,   0.9f, 0.2f, 0.85f);

    // ── Output dimensions (computed by chaining transforms) ───────────────────

    /// <summary>Width of the pixel array returned by <see cref="Capture"/>.</summary>
    public int OutputWidth  => ChainSize().X;
    /// <summary>Height of the pixel array returned by <see cref="Capture"/>.</summary>
    public int OutputHeight => ChainSize().Y;
    /// <summary>Channel count (1 = grayscale, 3 = RGB).</summary>
    public int OutputChannels
    {
        get
        {
            var ch = 3; // viewport always starts as RGB
            foreach (var r in _transforms)
                if (r is RLCameraTransform t) ch = t.ComputeOutputChannels(ch);
            return ch;
        }
    }

    private Vector2I ChainSize()
    {
        var size = _renderSize;
        foreach (var r in _transforms)
            if (r is RLCameraTransform t) size = t.ComputeOutputSize(size);
        return size;
    }

    /// <summary>
    /// Final post-transform texture used by camera debug overlay.
    /// Falls back to raw viewport texture until the first transformed frame is available.
    /// </summary>
    internal Texture2D? DebugTexture
    {
        get
        {
            _debugTextureRequested = true;
            if (_finalDebugTexture is not null)
                return _finalDebugTexture;

            return _viewport?.GetTexture();
        }
    }

    // ── Godot lifecycle ──────────────────────────────────────────────────────

    public override void _Ready()
    {
        if (DisplayServer.GetName() == "headless")
        {
            // Headless workers have no renderer — SubViewport produces nothing useful.
            // Capture() will return a correctly-sized zero array so observation size stays consistent.
            GD.PushWarning("[RLCameraSensor2D] Running headless — camera capture disabled. " +
                           "Disable distributed workers when training with camera sensors.");
            return;
        }
        BuildViewport();
    }

    public override void _Process(double delta)
    {
        if (_camera is null) return;
        _camera.GlobalPosition = GlobalPosition;
        _camera.GlobalRotation = GlobalRotation;
        _camera.Zoom           = new Vector2(_zoom, _zoom);

        // Do the GPU readback here (once per rendered frame) and cache the result.
        // Capture() then returns the cache — safe to call many times per physics step.
        _cachedPixels = ReadPixelsFromViewport();

        if (Engine.IsEditorHint())
            QueueRedraw();
    }

    public override void _Draw()
    {
        if (!Engine.IsEditorHint()) return;
        if (_zoom <= 0f) return;

        var renderW    = _renderSize.X / _zoom;
        var renderH    = _renderSize.Y / _zoom;
        var renderRect = new Rect2(-renderW * 0.5f, -renderH * 0.5f, renderW, renderH);

        DrawRect(renderRect, CRenderFill);
        DrawRect(renderRect, CRenderBorder, false, 1.5f);

        // Corner tick marks.
        var tick = Mathf.Min(renderW, renderH) * 0.08f;
        DrawLine(renderRect.Position,                           renderRect.Position + new Vector2(tick, 0),          CRenderBorder, 1.5f);
        DrawLine(renderRect.Position,                           renderRect.Position + new Vector2(0, tick),          CRenderBorder, 1.5f);
        DrawLine(renderRect.Position + new Vector2(renderW, 0), renderRect.Position + new Vector2(renderW - tick, 0), CRenderBorder, 1.5f);
        DrawLine(renderRect.Position + new Vector2(renderW, 0), renderRect.Position + new Vector2(renderW, tick),     CRenderBorder, 1.5f);
        DrawLine(renderRect.End,                                renderRect.End - new Vector2(tick, 0),               CRenderBorder, 1.5f);
        DrawLine(renderRect.End,                                renderRect.End - new Vector2(0, tick),               CRenderBorder, 1.5f);
        DrawLine(renderRect.Position + new Vector2(0, renderH), renderRect.Position + new Vector2(0, renderH - tick), CRenderBorder, 1.5f);
        DrawLine(renderRect.Position + new Vector2(0, renderH), renderRect.Position + new Vector2(tick, renderH),     CRenderBorder, 1.5f);

        // Centre crosshair.
        var cross = Mathf.Min(renderW, renderH) * 0.04f;
        DrawLine(new Vector2(-cross, 0), new Vector2(cross, 0), CRenderBorder, 1f);
        DrawLine(new Vector2(0, -cross), new Vector2(0, cross), CRenderBorder, 1f);

        // Yellow crop rect — drawn if an RLCropTransform is anywhere in the chain.
        foreach (var r in _transforms)
        {
            if (r is RLCropTransform crop)
            {
                var cropRect = new Rect2(
                    renderRect.Position.X + crop.Offset.X / _zoom,
                    renderRect.Position.Y + crop.Offset.Y / _zoom,
                    crop.Size.X / _zoom,
                    crop.Size.Y / _zoom);
                DrawRect(cropRect, CCropBorder, false, 1.5f);
                break; // only draw the first crop found
            }
        }
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// <summary>
    /// Returns the most recently captured and transformed pixel bytes. The GPU readback
    /// and transform pipeline runs once per rendered frame in <c>_Process</c> and is cached —
    /// calling this multiple times per physics step is free.
    /// Length = OutputWidth × OutputHeight × OutputChannels.
    /// </summary>
    public byte[] Capture()
    {
        // Headless: no viewport — return correctly-sized zeros.
        if (_viewport is null)
            return new byte[OutputWidth * OutputHeight * OutputChannels];

        // Return frame cache. If _Process hasn't run yet, do a one-time readback.
        return _cachedPixels ?? ReadPixelsFromViewport() ?? new byte[OutputWidth * OutputHeight * OutputChannels];
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// <summary>Runs the GPU readback and applies the transform pipeline.</summary>
    private byte[]? ReadPixelsFromViewport()
    {
        if (_viewport is null) return null;
        if (_dirty) RebuildCache();

        var image = _viewport.GetTexture().GetImage();
        if (image is null) return null;

        image.Convert(Image.Format.Rgb8); // normalise to RGB before transforms

        foreach (var r in _transforms)
            if (r is RLCameraTransform t) image = t.Apply(image);

        if (_debugTextureRequested)
        {
            _finalDebugTexture ??= ImageTexture.CreateFromImage(image);
            _finalDebugTexture.Update(image);
        }

        return image.GetData();
    }

    private void BuildViewport()
    {
        _viewport = new SubViewport
        {
            Size                   = _renderSize,
            RenderTargetUpdateMode = SubViewport.UpdateMode.Always,
            TransparentBg          = false,
        };
        AddChild(_viewport);
        _viewport.World2D = GetViewport().World2D; // Share the main scene's World2D so the internal camera sees scene content.

        _camera = new Camera2D { Enabled = true };
        _viewport.AddChild(_camera);

        RebuildCache();
    }

    private void RebuildCache()
    {
        _dirty = false;
        if (_viewport is not null)
            _viewport.Size = _renderSize;
    }
}
