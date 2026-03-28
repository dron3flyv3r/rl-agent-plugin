using System;
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
/// <b>Background fill</b><br/>
/// Assign a <see cref="RLCameraBackground"/> resource to <see cref="Background"/> to control
/// what is rendered when the camera extends beyond scene content. Built-in strategies:
/// <list type="bullet">
///   <item><see cref="RLSolidBackground"/> — flat solid colour.</item>
///   <item><see cref="RLMirrorBackground"/> — reflects scene content at the edge.</item>
///   <item><see cref="RLExtendBackground"/> — clamps/stretches the nearest edge pixel outward.</item>
/// </list>
/// When no background is set the viewport uses its default clear colour (existing behaviour).
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

    /// <summary>
    /// When true, monitors frame-to-frame pixel change after each GPU readback.
    /// If the camera output is identical for <c>MotionFrozenWarnFrames</c> consecutive
    /// captured frames it fires a warning via <see cref="OnValidationLog"/> (visible on
    /// the master console when running distributed workers).
    ///
    /// Enable this on any sensor whose scene is expected to be in motion during training.
    /// A frozen warning means physics may be stalled, the simulation is paused, or the
    /// camera is not seeing the scene (e.g. still running headless without a renderer).
    /// </summary>
    [Export]
    public bool ValidateMotion { get; set; } = false;

    /// <summary>
    /// How many consecutive captured frames with zero pixel change trigger a frozen warning.
    /// Defaults to 60 — roughly one second at 60 fps.
    /// Increase this if your scene has legitimate static moments at episode start.
    /// </summary>
    [Export(PropertyHint.Range, "10,600,1")]
    public int MotionFrozenWarnFrames { get; set; } = 60;

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
    /// Optional background fill strategy applied when the camera extends beyond scene content.
    /// When set, the viewport renders with a transparent background and any pixel with
    /// alpha == 0 is filled by this strategy before the transform pipeline runs.
    /// When null, the viewport uses its default clear colour (no special handling).
    /// </summary>
    [Export(PropertyHint.ResourceType, nameof(RLCameraBackground))]
    public RLCameraBackground? Background
    {
        get => _background;
        set { _background = value; _dirty = true; }
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

    // ── Static events ─────────────────────────────────────────────────────────

    /// <summary>
    /// Fired for each capture-validation log line. On worker processes TrainingBootstrap
    /// subscribes to this and routes the message back to the master via TCP so it appears
    /// in the master console. Falls back to <c>GD.Print</c> when no subscriber is attached.
    /// </summary>
    public static event Action<string>? OnValidationLog;

    // ── Private state ────────────────────────────────────────────────────────

    private Vector2I         _renderSize  = new(128, 128);
    private float            _zoom        = 1.0f;
    private RLCameraBackground? _background = null;
    private Array<Resource>  _transforms  = [];

    private SubViewport? _viewport;
    private Camera2D?    _camera;

    // Pixel cache — populated lazily when Capture() is first called after each rendered frame.
    private byte[]?       _cachedPixels;
    private bool          _pixelsDirty = true;   // true whenever a new frame has been rendered
    private ImageTexture? _finalDebugTexture;
    private bool          _debugTextureRequested;
    private int           _captureValidationCount; // counts captures remaining for startup log

    // Motion validation state.
    private byte[]? _motionPrevPixels;       // previous frame's pixels for diff comparison
    private int     _motionWarmupRemaining;  // skip this many captures before checking
    private int     _frozenFrameCount;       // consecutive captures with zero pixel change
    private int     _frozenWarnCooldown;     // suppress repeated warnings for this many captures
    private const int FrozenWarnCooldownFrames = 300;

    // Observation-staleness stats: count unique GPU readbacks vs total Capture() calls.
    // Reported every _statsWindowSize calls so each report ≈ one rollout window.
    private int _statsTotalCaptures;
    private int _statsUniqueFrames;
    private const int StatsWindowSize = 256;

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
                           "Enable WorkersRequireRenderer on RLDistributedConfig to launch workers with a renderer.");
            return;
        }
        BuildViewport();
        _captureValidationCount  = 1;  // log the first capture to confirm real pixels
        _motionWarmupRemaining   = 30; // skip first 30 frames while physics settles
    }

    public override void _Process(double delta)
    {
        if (_camera is null) return;
        _camera.GlobalPosition = GlobalPosition;
        _camera.GlobalRotation = GlobalRotation;
        _camera.Zoom           = new Vector2(_zoom, _zoom);

        // Sync TransparentBg with whether a background strategy is configured.
        if (_dirty) RebuildCache();

        // Mark that a new frame has been rendered. The actual GPU readback is deferred
        // to the first Capture() call so we never stall the GPU when no agent needs pixels.
        _pixelsDirty = true;

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

        // Only do the GPU readback when a new frame has been rendered since the last call.
        // Multiple Capture() calls within the same physics step return the cached result for free.
        var isNewFrame = _pixelsDirty || _cachedPixels is null;
        if (isNewFrame)
        {
            _pixelsDirty  = false;
            _cachedPixels = ReadPixelsFromViewport() ?? new byte[OutputWidth * OutputHeight * OutputChannels];

            if (_captureValidationCount > 0)
            {
                _captureValidationCount--;
                var nonZero = 0;
                foreach (var b in _cachedPixels) if (b != 0) nonZero++;
                var pct = nonZero * 100f / _cachedPixels.Length;
                var msg = $"[RLCameraSensor2D] Capture validation ({Name}): " +
                          $"{nonZero}/{_cachedPixels.Length} non-zero bytes ({pct:F1}%). " +
                          (nonZero == 0 ? "WARNING: all zeros — renderer may not be active." : "OK.");
                if (OnValidationLog is not null) OnValidationLog(msg);
                else GD.Print(msg);
            }

            if (ValidateMotion)
                CheckMotion(_cachedPixels);
        }

        if (ValidateMotion)
        {
            if (isNewFrame) _statsUniqueFrames++;
            _statsTotalCaptures++;

            if (_statsTotalCaptures >= StatsWindowSize)
            {
                var pct = _statsUniqueFrames * 100f / _statsTotalCaptures;
                var msg = $"[RLCameraSensor2D] Observation staleness ({Name}): " +
                          $"{_statsUniqueFrames} unique frames in {_statsTotalCaptures} captures ({pct:F1}%). " +
                          (_statsUniqueFrames <= 1
                              ? "WARNING: near-zero unique frames — observations are effectively static."
                              : pct < 10f
                                  ? "WARNING: very stale observations — many steps share the same frame."
                                  : "OK.");
                if (OnValidationLog is not null) OnValidationLog(msg);
                else GD.Print(msg);
                _statsTotalCaptures = 0;
                _statsUniqueFrames  = 0;
            }
        }

        return _cachedPixels!;
    }

    // ── Motion validation ─────────────────────────────────────────────────────

    private void CheckMotion(byte[] pixels)
    {
        if (_motionWarmupRemaining > 0)
        {
            _motionWarmupRemaining--;
            _motionPrevPixels = (byte[])pixels.Clone();
            return;
        }

        if (_motionPrevPixels is null)
        {
            _motionPrevPixels = (byte[])pixels.Clone();
            return;
        }

        var diff = 0;
        for (var i = 0; i < pixels.Length; i++)
            diff += Math.Abs(pixels[i] - _motionPrevPixels[i]);

        System.Array.Copy(pixels, _motionPrevPixels, pixels.Length);

        if (diff > 0)
        {
            _frozenFrameCount   = 0;
            _frozenWarnCooldown = 0;
            return;
        }

        _frozenFrameCount++;

        if (_frozenWarnCooldown > 0)
        {
            _frozenWarnCooldown--;
            return;
        }

        if (_frozenFrameCount >= MotionFrozenWarnFrames)
        {
            var msg = $"[RLCameraSensor2D] Motion validation ({Name}): " +
                      $"no pixel change for {_frozenFrameCount} consecutive frames — " +
                      "physics may be stalled or camera stuck on a single frame.";
            if (OnValidationLog is not null) OnValidationLog(msg);
            else GD.PushWarning(msg);
            _frozenWarnCooldown = FrozenWarnCooldownFrames;
        }
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// <summary>Runs the GPU readback, applies background fill, then the transform pipeline.</summary>
    private byte[]? ReadPixelsFromViewport()
    {
        if (_viewport is null) return null;

        var image = _viewport.GetTexture().GetImage();
        if (image is null) return null;

        if (_background is not null)
        {
            // Background strategies operate on RGBA so transparent pixels are identifiable.
            image.Convert(Image.Format.Rgba8);
            image = _background.Apply(image);
        }

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
            TransparentBg          = _background is not null,
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
        if (_viewport is null) return;
        _viewport.Size          = _renderSize;
        _viewport.TransparentBg = _background is not null;
    }
}
