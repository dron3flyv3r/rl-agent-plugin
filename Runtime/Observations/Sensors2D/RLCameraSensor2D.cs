using Godot;

namespace RlAgentPlugin.Runtime;

/// <summary>
/// A camera sensor node for 2D scenes. Place it anywhere in your scene and position/rotate
/// it like a regular Camera2D. Call <c>buffer.AddImage("name", this)</c> from your agent's
/// <c>CollectObservations</c> override to feed the captured pixels into the observation.
///
/// Internally the sensor owns a <see cref="SubViewport"/> that renders the world from a
/// mirrored <see cref="Camera2D"/>. Each <see cref="Capture"/> call reads back the rendered
/// texture, optionally crops it, and optionally converts it to grayscale.
///
/// Editor overlay: the white rectangle shows the full render area; the yellow rectangle
/// shows the crop region that is actually fed to the network.
/// </summary>
[Tool]
[GlobalClass]
public partial class RLCameraSensor2D : Node2D
{
    // ── Exports ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Optional CNN/MLP encoder config for this sensor's image stream.
    /// Assign this to configure how this sensor's image stream is encoded by the network.
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

    [Export]
    public Vector2I CropSize
    {
        get => _cropSize;
        set { _cropSize = value; _dirty = true; QueueRedraw(); }
    }

    /// <summary>
    /// Pixel offset of the crop window from the top-left of the rendered image.
    /// (0,0) = top-left. To centre: (RenderSize - CropSize) / 2.
    /// </summary>
    [Export]
    public Vector2I CropOffset
    {
        get => _cropOffset;
        set { _cropOffset = value; _dirty = true; QueueRedraw(); }
    }

    /// <summary>When true the output has 1 channel (luminance). When false: 3 channels (RGB).</summary>
    [Export]
    public bool Grayscale
    {
        get => _grayscale;
        set { _grayscale = value; _dirty = true; }
    }

    /// <summary>Zoom of the internal camera. 1 = world units match viewport pixels.</summary>
    [Export]
    public float Zoom
    {
        get => _zoom;
        set { _zoom = Mathf.Max(0.001f, value); _dirty = true; QueueRedraw(); }
    }

    // ── Private state ────────────────────────────────────────────────────────

    private Vector2I _renderSize = new(128, 128);
    private Vector2I _cropSize   = new(64, 64);
    private Vector2I _cropOffset = new(0, 0);
    private bool     _grayscale  = true;
    private float    _zoom       = 1.0f;

    private SubViewport? _viewport;
    private Camera2D?    _camera;

    // Pixel cache — refreshed once per rendered frame in _Process, not per physics step.
    private byte[]? _cachedPixels;

    private bool _dirty    = true;
    private int  _cropW;
    private int  _cropH;
    private int  _channels;

    // Overlay colours.
    private static readonly Color CRenderFill   = new(0.5f, 0.8f, 1f,   0.06f);
    private static readonly Color CRenderBorder = new(1f,   1f,   1f,   0.35f);
    private static readonly Color CCropBorder   = new(1f,   0.9f, 0.2f, 0.85f);

    /// <summary>Width of the pixel array returned by <see cref="Capture"/>.</summary>
    public int OutputWidth    => _cropSize.X;
    /// <summary>Height of the pixel array returned by <see cref="Capture"/>.</summary>
    public int OutputHeight   => _cropSize.Y;
    /// <summary>Channel count (1 = grayscale, 3 = RGB).</summary>
    public int OutputChannels => _grayscale ? 1 : 3;

    /// <summary>The raw SubViewport texture — used by the camera debug overlay.</summary>
    internal ViewportTexture? ViewportTexture => _viewport?.GetTexture();

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

        // Crop region.
        var cropRect = new Rect2(
            renderRect.Position.X + _cropOffset.X / _zoom,
            renderRect.Position.Y + _cropOffset.Y / _zoom,
            _cropSize.X / _zoom,
            _cropSize.Y / _zoom);
        DrawRect(cropRect, CCropBorder, false, 1.5f);

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
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// <summary>
    /// Returns the most recently captured pixel bytes. The GPU readback happens once per
    /// rendered frame in <c>_Process</c> and is cached — calling this multiple times per
    /// physics step is free. Length = OutputWidth × OutputHeight × OutputChannels.
    /// </summary>
    public byte[] Capture()
    {
        if (_dirty) RebuildCache();

        // Headless: no viewport — return correctly-sized zeros.
        if (_viewport is null)
            return new byte[_cropW * _cropH * _channels];

        // Return frame cache. If _Process hasn't run yet, do a one-time readback.
        return _cachedPixels ?? ReadPixelsFromViewport() ?? new byte[_cropW * _cropH * _channels];
    }

    /// <summary>Does the actual GPU readback + crop + channel conversion.</summary>
    private byte[]? ReadPixelsFromViewport()
    {
        if (_viewport is null) return null;
        if (_dirty) RebuildCache();

        var tex   = _viewport.GetTexture();
        var image = tex.GetImage();
        if (image is null) return null;

        image.Convert(Image.Format.Rgb8);

        var imgW    = image.GetWidth();
        var imgH    = image.GetHeight();
        var offX    = Mathf.Clamp(_cropOffset.X, 0, Mathf.Max(0, imgW - _cropW));
        var offY    = Mathf.Clamp(_cropOffset.Y, 0, Mathf.Max(0, imgH - _cropH));
        var actualW = Mathf.Min(_cropW, imgW - offX);
        var actualH = Mathf.Min(_cropH, imgH - offY);

        var output = new byte[actualW * actualH * _channels];
        var idx    = 0;

        for (var row = 0; row < actualH; row++)
        {
            for (var col = 0; col < actualW; col++)
            {
                var pixel = image.GetPixel(offX + col, offY + row);
                if (_channels == 1)
                    output[idx++] = (byte)(pixel.R * 76 + pixel.G * 150 + pixel.B * 29);
                else
                {
                    output[idx++] = (byte)(pixel.R * 255f);
                    output[idx++] = (byte)(pixel.G * 255f);
                    output[idx++] = (byte)(pixel.B * 255f);
                }
            }
        }

        return output;
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    private void BuildViewport()
    {
        _viewport = new SubViewport
        {
            Size                   = _renderSize,
            RenderTargetUpdateMode = SubViewport.UpdateMode.Always,
            TransparentBg          = false,
        };
        AddChild(_viewport);

        _camera = new Camera2D { Enabled = true };
        _viewport.AddChild(_camera);

        RebuildCache();
    }

    private void RebuildCache()
    {
        _cropW    = Mathf.Max(1, _cropSize.X);
        _cropH    = Mathf.Max(1, _cropSize.Y);
        _channels = _grayscale ? 1 : 3;
        _dirty    = false;

        if (_viewport is not null)
            _viewport.Size = _renderSize;
    }
}
