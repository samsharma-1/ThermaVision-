import {
  useState,
  useRef,
  useCallback,
  useEffect,
} from "react";
import {
  Upload,
  Globe,
  FileText,
  Settings,
  Bell,
  User,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Download,
  CheckCircle,
  Activity,
  Layers,
  Shield,
  ArrowRight,
  Eye,
  RefreshCw,
  Info,
  Zap,
  Target,
  Cpu,
  GitBranch,
  TrendingUp,
  BarChart2,
  Navigation,
  ChevronRight,
  Clock,
  MapPin,
  Database,
} from "lucide-react";
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  ResponsiveContainer,
} from "recharts";

// ─── Data ────────────────────────────────────────────────────────────────────

const DETECTED_OBJECTS = [
  {
    id: "roads",
    label: "Road Networks",
    count: "12 segments",
    confidence: 94,
    color: "#f59e0b",
  },
  {
    id: "buildings",
    label: "Urban Structures",
    count: "47 buildings",
    confidence: 89,
    color: "#60a5fa",
  },
  {
    id: "veg",
    label: "Vegetation Cover",
    count: "8.3 km²",
    confidence: 97,
    color: "#34d399",
  },
  {
    id: "vehicles",
    label: "Vehicles / Assets",
    count: "23 units",
    confidence: 78,
    color: "#f87171",
  },
  {
    id: "water",
    label: "Water Bodies",
    count: "2 bodies",
    confidence: 95,
    color: "#38bdf8",
  },
];

const RADAR_DATA = [
  { subject: "Roads", value: 94 },
  { subject: "Buildings", value: 89 },
  { subject: "Vegetation", value: 97 },
  { subject: "Vehicles", value: 78 },
  { subject: "Water", value: 95 },
];

const PIPELINE_STEPS = [
  {
    id: 1,
    name: "IR Input",
    desc: "Raw sensor data ingestion",
    icon: Upload,
    color: "#4a7aaa",
  },
  {
    id: 2,
    name: "Registration",
    desc: "Geometric alignment & co-registration",
    icon: Navigation,
    color: "#4a9fff",
  },
  {
    id: 3,
    name: "SSL Learning",
    desc: "MAE pre-training on unlabeled data",
    icon: Cpu,
    color: "#00c8ff",
  },
  {
    id: 4,
    name: "Feature Ext.",
    desc: "Multi-scale encoder representations",
    icon: GitBranch,
    color: "#00d4ff",
  },
  {
    id: 5,
    name: "Enhancement",
    desc: "U-Net super-resolution processing",
    icon: Zap,
    color: "#00ff9d",
  },
  {
    id: 6,
    name: "RGB Synthesis",
    desc: "GAN-based IR-to-RGB translation",
    icon: Eye,
    color: "#00ee88",
  },
  {
    id: 7,
    name: "Object Det.",
    desc: "YOLOv8 multi-class detection",
    icon: Target,
    color: "#ff8c00",
  },
  {
    id: 8,
    name: "Confidence",
    desc: "Bayesian uncertainty quantification",
    icon: Shield,
    color: "#ff6a00",
  },
  {
    id: 9,
    name: "Decision Sup.",
    desc: "Analyst-ready report generation",
    icon: FileText,
    color: "#ff4040",
  },
];

const TECH_BADGES = [
  { name: "Python 3.11", color: "#3b82f6" },
  { name: "PyTorch 2.3", color: "#ef4444" },
  { name: "OpenCV 4.9", color: "#22c55e" },
  { name: "YOLOv8", color: "#f59e0b" },
  { name: "Masked Autoencoder", color: "#8b5cf6" },
  { name: "U-Net", color: "#06b6d4" },
  { name: "GAN", color: "#ec4899" },
  { name: "Rasterio", color: "#14b8a6" },
  { name: "GDAL 3.8", color: "#f97316" },
  { name: "CUDA 12.2", color: "#84cc16" },
];

const STATS = [
  {
    label: "PSNR",
    value: "34.7",
    unit: "dB",
    icon: TrendingUp,
  },
  { label: "SSIM", value: "0.9312", unit: "", icon: BarChart2 },
  { label: "LPIPS", value: "0.0847", unit: "", icon: Activity },
  {
    label: "Objects Detected",
    value: "93",
    unit: "total",
    icon: Target,
  },
  {
    label: "Processing Time",
    value: "2.34",
    unit: "sec",
    icon: Clock,
  },
  {
    label: "Model Version",
    value: "IRIS v2.4",
    unit: "",
    icon: Cpu,
  },
];

// ─── Image Comparison Slider ─────────────────────────────────────────────────

function ImageSlider() {
  const [pos, setPos] = useState(42);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const clamp = (v: number) => Math.max(2, Math.min(98, v));

  const applyPos = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const r = containerRef.current.getBoundingClientRect();
    setPos(clamp(((clientX - r.left) / r.width) * 100));
  }, []);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (dragging.current) applyPos(e.clientX);
    };
    const onUp = () => {
      dragging.current = false;
    };
    const onTouch = (e: TouchEvent) => {
      if (dragging.current) applyPos(e.touches[0].clientX);
    };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    window.addEventListener("touchmove", onTouch, {
      passive: true,
    });
    window.addEventListener("touchend", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("touchmove", onTouch);
      window.removeEventListener("touchend", onUp);
    };
  }, [applyPos]);

  const IR_IMG =
    "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=1000&h=640&fit=crop&auto=format";
  const RGB_IMG =
    "https://images.unsplash.com/photo-1446776877081-d282a0f896e2?w=1000&h=640&fit=crop&auto=format";

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full min-h-[280px] rounded-xl overflow-hidden bg-[#020408] select-none"
    >
      {/* Corner decorations */}
      <div className="absolute top-0 left-0 w-8 h-8 border-l-2 border-t-2 border-cyan-400/40 z-30 rounded-tl-xl pointer-events-none" />
      <div className="absolute top-0 right-0 w-8 h-8 border-r-2 border-t-2 border-cyan-400/40 z-30 rounded-tr-xl pointer-events-none" />
      <div className="absolute bottom-0 left-0 w-8 h-8 border-l-2 border-b-2 border-cyan-400/40 z-30 rounded-bl-xl pointer-events-none" />
      <div className="absolute bottom-0 right-0 w-8 h-8 border-r-2 border-b-2 border-cyan-400/40 z-30 rounded-br-xl pointer-events-none" />

      {/* Labels */}
      <div
        className="absolute top-3 left-3 z-20 px-2 py-1 rounded text-[10px] font-mono tracking-wider text-orange-300 border border-orange-400/30"
        style={{ background: "rgba(0,0,0,0.65)" }}
      >
        IR — ORIGINAL
      </div>
      <div
        className="absolute top-3 right-3 z-20 px-2 py-1 rounded text-[10px] font-mono tracking-wider text-cyan-300 border border-cyan-400/30"
        style={{ background: "rgba(0,0,0,0.65)" }}
      >
        RGB — ENHANCED
      </div>

      {/* Scan line overlay */}
      <div
        className="absolute inset-0 z-10 pointer-events-none"
        style={{
          backgroundImage:
            "repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,212,255,0.012) 3px, rgba(0,212,255,0.012) 4px)",
        }}
      />

      {/* IR base image */}
      <img
        src={IR_IMG}
        alt="Original infrared satellite imagery of Earth"
        draggable={false}
        className="absolute inset-0 w-full h-full object-cover"
        style={{
          filter:
            "grayscale(100%) contrast(1.35) brightness(0.6) sepia(15%)",
        }}
      />

      {/* RGB enhanced, clipped to slider left */}
      <div
        className="absolute inset-0"
        style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
      >
        <img
          src={RGB_IMG}
          alt="AI-enhanced RGB satellite imagery"
          draggable={false}
          className="absolute inset-0 w-full h-full object-cover"
        />
        {/* Detection overlays on enhanced side */}
        <div className="absolute inset-0">
          {/* Road */}
          <div
            className="absolute"
            style={{ top: "68%", left: "15%", width: "38%" }}
          >
            <div className="h-0.5 w-full bg-yellow-400/80 shadow-[0_0_6px_rgba(251,191,36,0.8)]" />
            <div className="absolute -top-4 left-1/3 text-[9px] text-yellow-300 font-mono whitespace-nowrap bg-black/50 px-1 rounded">
              ROAD 94%
            </div>
          </div>
          {/* Buildings cluster */}
          <div
            className="absolute"
            style={{ top: "28%", left: "22%" }}
          >
            <div className="w-12 h-8 border border-blue-400/70 rounded-sm shadow-[0_0_8px_rgba(96,165,250,0.5)] bg-blue-400/5" />
            <div className="absolute -top-4 left-0 text-[9px] text-blue-300 font-mono whitespace-nowrap bg-black/50 px-1 rounded">
              BLD 89%
            </div>
          </div>
          {/* Vegetation */}
          <div
            className="absolute"
            style={{ top: "52%", left: "8%" }}
          >
            <div className="w-14 h-10 rounded-full border border-green-400/60 bg-green-400/10 shadow-[0_0_10px_rgba(52,211,153,0.4)]" />
            <div className="absolute -top-4 left-1 text-[9px] text-green-300 font-mono whitespace-nowrap bg-black/50 px-1 rounded">
              VEG 97%
            </div>
          </div>
          {/* Vehicle */}
          <div
            className="absolute"
            style={{ top: "45%", left: "48%" }}
          >
            <div className="w-2.5 h-2.5 rounded-sm border border-red-400 bg-red-400/30 shadow-[0_0_6px_rgba(248,113,113,0.7)]" />
            <div className="absolute -top-4 -left-2 text-[9px] text-red-300 font-mono whitespace-nowrap bg-black/50 px-1 rounded">
              VEH 78%
            </div>
          </div>
          {/* Water body */}
          <div
            className="absolute"
            style={{
              top: "18%",
              left: "55%",
              width: "16%",
              height: "11%",
            }}
          >
            <div className="w-full h-full rounded-full border border-sky-400/50 bg-sky-400/15 shadow-[0_0_12px_rgba(56,189,248,0.4)]" />
            <div className="absolute -top-4 left-0 text-[9px] text-sky-300 font-mono whitespace-nowrap bg-black/50 px-1 rounded">
              WATER 95%
            </div>
          </div>
        </div>
      </div>

      {/* Slider handle */}
      <div
        className="absolute top-0 bottom-0 z-20 cursor-ew-resize"
        style={{
          left: `${pos}%`,
          transform: "translateX(-50%)",
        }}
        onMouseDown={() => {
          dragging.current = true;
        }}
        onTouchStart={() => {
          dragging.current = true;
        }}
      >
        <div
          className="h-full w-px"
          style={{
            background:
              "linear-gradient(180deg, transparent 0%, #00d4ff 10%, #00d4ff 90%, transparent 100%)",
            boxShadow: "0 0 10px rgba(0,212,255,0.7)",
          }}
        />
        <div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-9 h-9 rounded-full flex items-center justify-center border-2 border-cyan-400"
          style={{
            background: "#050d1e",
            boxShadow: "0 0 16px rgba(0,212,255,0.6)",
          }}
        >
          <div className="flex gap-0.5">
            <div className="w-0.5 h-3.5 rounded-full bg-cyan-400" />
            <div className="w-0.5 h-3.5 rounded-full bg-cyan-400" />
          </div>
        </div>
      </div>

      {/* Bottom coordinate bar */}
      <div
        className="absolute bottom-3 left-1/2 -translate-x-1/2 z-20 px-3 py-1 rounded text-[9px] font-mono text-cyan-400/70 border border-cyan-400/15 whitespace-nowrap"
        style={{ background: "rgba(0,0,0,0.6)" }}
      >
        21.4°N · 78.9°E · WGS84 / UTM 44N · INSAT-3DR / 01 JUL
        2026 08:14 UTC
      </div>
    </div>
  );
}

// ─── Confidence Heatmap View ──────────────────────────────────────────────────

function HeatmapView() {
  const [opacity, setOpacity] = useState(72);

  const o = opacity / 100;
  const heatStyle = {
    background: `
      radial-gradient(ellipse 18% 14% at 26% 30%, rgba(0,255,100,${o}) 0%, transparent 100%),
      radial-gradient(ellipse 13% 11% at 50% 44%, rgba(0,255,100,${o * 0.9}) 0%, transparent 100%),
      radial-gradient(ellipse 10% 9%  at 68% 26%, rgba(0,255,100,${o}) 0%, transparent 100%),
      radial-gradient(ellipse 15% 12% at 39% 70%, rgba(255,200,0,${o * 0.85}) 0%, transparent 100%),
      radial-gradient(ellipse 11% 9%  at 20% 73%, rgba(255,200,0,${o * 0.8}) 0%, transparent 100%),
      radial-gradient(ellipse 9%  7%  at 74% 72%, rgba(255,80,0,${o * 0.7}) 0%, transparent 100%),
      radial-gradient(ellipse 7%  6%  at 86% 54%, rgba(255,30,30,${o * 0.65}) 0%, transparent 100%),
      radial-gradient(ellipse 12% 10% at 12% 46%, rgba(0,255,100,${o * 0.85}) 0%, transparent 100%)
    `,
  };

  return (
    <div className="p-5 h-full flex flex-col gap-4">
      <div className="flex items-center justify-between flex-shrink-0">
        <div>
          <h2 className="text-base font-semibold text-foreground tracking-wide">
            Confidence Heatmap
          </h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            Spatial confidence distribution across detected
            regions
          </p>
        </div>
        <div className="flex items-center gap-5 text-[11px] font-mono">
          {[
            { label: "High >85%", dot: "bg-green-400" },
            { label: "Medium 65–85%", dot: "bg-yellow-400" },
            { label: "Low <65%", dot: "bg-red-500" },
          ].map((l) => (
            <div
              key={l.label}
              className="flex items-center gap-1.5"
            >
              <div
                className={`w-2.5 h-2.5 rounded-full ${l.dot}`}
              />
              <span className="text-muted-foreground">
                {l.label}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Opacity control */}
      <div className="flex items-center gap-3 flex-shrink-0">
        <span className="text-xs text-muted-foreground font-mono w-28">
          Overlay: {opacity}%
        </span>
        <input
          type="range"
          min={0}
          max={100}
          value={opacity}
          onChange={(e) => setOpacity(Number(e.target.value))}
          className="flex-1 accent-cyan-400 h-1 rounded"
        />
      </div>

      {/* Map */}
      <div className="relative flex-1 rounded-xl overflow-hidden border border-border min-h-[280px]">
        <img
          src="https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=1200&h=700&fit=crop&auto=format"
          alt="Satellite imagery base for heatmap"
          className="absolute inset-0 w-full h-full object-cover"
          style={{ filter: "grayscale(50%) brightness(0.6)" }}
        />
        {/* Heatmap overlay */}
        <div
          className="absolute inset-0"
          style={{ ...heatStyle, mixBlendMode: "screen" }}
        />
        {/* Grid reticle */}
        <div
          className="absolute inset-0 pointer-events-none"
          style={{
            backgroundImage:
              "linear-gradient(rgba(0,212,255,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(0,212,255,0.04) 1px, transparent 1px)",
            backgroundSize: "48px 48px",
          }}
        />
        {/* Coordinate labels */}
        <div className="absolute bottom-3 right-3 text-[9px] font-mono text-cyan-400/60 bg-black/50 px-2 py-1 rounded border border-cyan-400/10">
          Scene: INSAT3D-20260701-0814 · 847×623 km
        </div>
        {/* Zone annotations */}
        <div
          className="absolute"
          style={{ top: "18%", left: "18%" }}
        >
          <div className="text-[9px] font-mono text-green-300 bg-black/60 px-1 rounded border border-green-400/20">
            Zone A — 94%
          </div>
        </div>
        <div
          className="absolute"
          style={{ top: "60%", left: "33%" }}
        >
          <div className="text-[9px] font-mono text-yellow-300 bg-black/60 px-1 rounded border border-yellow-400/20">
            Zone B — 78%
          </div>
        </div>
        <div
          className="absolute"
          style={{ top: "65%", left: "70%" }}
        >
          <div className="text-[9px] font-mono text-red-300 bg-black/60 px-1 rounded border border-red-400/20">
            Zone C — 55%
          </div>
        </div>
      </div>

      {/* Zone stats */}
      <div className="grid grid-cols-3 gap-3 flex-shrink-0">
        {[
          {
            label: "High Confidence",
            value: "5",
            sub: "62.5% of scene",
            color: "text-green-400",
            border: "border-green-400/20",
            bg: "bg-green-400/5",
          },
          {
            label: "Medium Confidence",
            value: "2",
            sub: "25.0% of scene",
            color: "text-yellow-400",
            border: "border-yellow-400/20",
            bg: "bg-yellow-400/5",
          },
          {
            label: "Low Confidence",
            value: "1",
            sub: "12.5% of scene",
            color: "text-red-400",
            border: "border-red-400/20",
            bg: "bg-red-400/5",
          },
        ].map((s) => (
          <div
            key={s.label}
            className={`rounded-xl p-4 text-center border ${s.border} ${s.bg}`}
          >
            <div
              className={`text-3xl font-bold font-mono ${s.color}`}
            >
              {s.value}
            </div>
            <div className="text-xs text-foreground mt-1">
              {s.label}
            </div>
            <div className="text-[10px] text-muted-foreground mt-0.5">
              {s.sub}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Analysis Report View ─────────────────────────────────────────────────────

function ReportView() {
  return (
    <div className="p-6 overflow-y-auto">
      <div className="max-w-3xl mx-auto space-y-5">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <div className="text-[10px] text-cyan-400 font-mono tracking-widest mb-1">
              MISSION REPORT — REF: IRIS-2026-0701-004
            </div>
            <h2 className="text-xl font-semibold text-foreground">
              Analysis Summary
            </h2>
            <p className="text-xs text-muted-foreground mt-1">
              Generated: 01 July 2026, 14:32:07 IST · Analyst:
              Dr. Priya Sharma, ISRO-SAC
            </p>
          </div>
          <div className="flex gap-2">
            <button className="flex items-center gap-1.5 px-3 py-2 bg-card border border-border rounded-lg text-xs text-foreground hover:border-cyan-400/30 transition-colors">
              <Download className="w-3.5 h-3.5 text-cyan-400" />{" "}
              Export PDF
            </button>
            <button className="flex items-center gap-1.5 px-3 py-2 bg-card border border-border rounded-lg text-xs text-foreground hover:border-green-400/30 transition-colors">
              <Download className="w-3.5 h-3.5 text-green-400" />{" "}
              Download JSON
            </button>
          </div>
        </div>

        {/* Mission metadata */}
        <div className="bg-card border border-border rounded-xl p-5">
          <h3 className="text-sm font-medium text-foreground mb-4 flex items-center gap-2">
            <Info className="w-4 h-4 text-cyan-400" /> Mission
            Metadata
          </h3>
          <div className="grid grid-cols-2 gap-x-10 gap-y-2.5 text-xs">
            {[
              ["Sensor", "INSAT-3DR Thermal IR Band 5"],
              [
                "Native Resolution",
                "4 km/pixel → 500 m resampled",
              ],
              ["Acquisition Time", "2026-07-01 08:14:22 UTC"],
              ["Scene Coverage", "847 × 623 km"],
              [
                "Centre Coords",
                "21.4°N, 78.9°E (Central India)",
              ],
              [
                "Processing Mode",
                "Automated AI + Analyst Review",
              ],
              ["Cloud Cover", "12.4% (partial obstruction)"],
              ["Data Quality", "94.1 / 100"],
            ].map(([k, v]) => (
              <div
                key={k}
                className="flex justify-between gap-3 border-b border-border pb-2"
              >
                <span className="text-muted-foreground">
                  {k}
                </span>
                <span className="text-foreground font-mono">
                  {v}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Detected features */}
        <div className="bg-card border border-border rounded-xl p-5">
          <h3 className="text-sm font-medium text-foreground mb-4 flex items-center gap-2">
            <Target className="w-4 h-4 text-cyan-400" />{" "}
            Detected Features
          </h3>
          <div className="space-y-3">
            {DETECTED_OBJECTS.map((obj) => (
              <div
                key={obj.id}
                className="flex items-center gap-3"
              >
                <div className="w-32 text-xs text-muted-foreground">
                  {obj.label}
                </div>
                <div className="flex-1 h-1.5 bg-[#0a1628] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${obj.confidence}%`,
                      backgroundColor: obj.color,
                    }}
                  />
                </div>
                <div
                  className="w-10 text-right text-xs font-mono"
                  style={{ color: obj.color }}
                >
                  {obj.confidence}%
                </div>
                <div className="w-24 text-right text-[11px] text-muted-foreground font-mono">
                  {obj.count}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Risk assessment */}
        <div className="bg-card border border-border rounded-xl p-5">
          <h3 className="text-sm font-medium text-foreground mb-4 flex items-center gap-2">
            <Shield className="w-4 h-4 text-orange-400" /> Risk
            Assessment
          </h3>
          <div className="grid grid-cols-3 gap-3">
            {[
              {
                label: "Flood Risk",
                level: "MODERATE",
                color: "text-yellow-400",
                border: "border-yellow-400/20",
                bg: "bg-yellow-400/5",
                detail:
                  "Elevated water body levels near 3 settlements",
              },
              {
                label: "Vegetation Stress",
                level: "LOW",
                color: "text-green-400",
                border: "border-green-400/20",
                bg: "bg-green-400/5",
                detail:
                  "NDVI indices within normal seasonal range",
              },
              {
                label: "Infrastructure",
                level: "NORMAL",
                color: "text-cyan-400",
                border: "border-cyan-400/20",
                bg: "bg-cyan-400/5",
                detail: "Road networks and structures intact",
              },
            ].map((r) => (
              <div
                key={r.label}
                className={`rounded-xl p-4 border ${r.border} ${r.bg}`}
              >
                <div className="text-[10px] text-muted-foreground mb-1">
                  {r.label}
                </div>
                <div
                  className={`text-base font-bold font-mono mb-2 ${r.color}`}
                >
                  {r.level}
                </div>
                <div className="text-[10px] text-muted-foreground leading-relaxed">
                  {r.detail}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Confidence metrics */}
        <div className="bg-card border border-border rounded-xl p-5">
          <h3 className="text-sm font-medium text-foreground mb-5 flex items-center gap-2">
            <Activity className="w-4 h-4 text-green-400" />{" "}
            Overall Confidence Metrics
          </h3>
          <div className="grid grid-cols-3 gap-4">
            {[
              {
                label: "Overall Reliability",
                value: "91.2%",
                color: "text-green-400",
              },
              {
                label: "Prediction Accuracy",
                value: "88.7%",
                color: "text-cyan-400",
              },
              {
                label: "Image Quality Index",
                value: "94.1",
                color: "text-blue-400",
              },
            ].map((m) => (
              <div key={m.label} className="text-center">
                <div
                  className={`text-4xl font-bold font-mono ${m.color}`}
                >
                  {m.value}
                </div>
                <div className="text-xs text-muted-foreground mt-2">
                  {m.label}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Model metrics */}
        <div className="bg-card border border-border rounded-xl p-5">
          <h3 className="text-sm font-medium text-foreground mb-4 flex items-center gap-2">
            <BarChart2 className="w-4 h-4 text-cyan-400" />{" "}
            Image Quality Metrics
          </h3>
          <div className="grid grid-cols-3 gap-4">
            {[
              {
                label: "PSNR",
                value: "34.7 dB",
                sub: "Peak Signal-to-Noise",
                good: true,
              },
              {
                label: "SSIM",
                value: "0.9312",
                sub: "Structural Similarity",
                good: true,
              },
              {
                label: "LPIPS",
                value: "0.0847",
                sub: "Perceptual Loss (lower↓)",
                good: false,
              },
            ].map((m) => (
              <div
                key={m.label}
                className="text-center p-3 rounded-lg bg-[#060e1c] border border-border"
              >
                <div
                  className={`text-2xl font-bold font-mono ${m.good ? "text-green-400" : "text-cyan-400"}`}
                >
                  {m.value}
                </div>
                <div className="text-xs text-foreground mt-1 font-medium">
                  {m.label}
                </div>
                <div className="text-[10px] text-muted-foreground">
                  {m.sub}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── App ─────────────────────────────────────────────────────────────────────

export default function App() {
  const [tab, setTab] = useState<
    "dashboard" | "heatmap" | "report"
  >("dashboard");
  const [uploadState, setUploadState] = useState<
    "idle" | "uploading" | "processing" | "complete"
  >("complete");
  const [uploadProgress, setUploadProgress] = useState(100);
  const [isDragOver, setIsDragOver] = useState(false);
  const [navActive, setNavActive] = useState("Dashboard");

  const simulateUpload = () => {
    if (
      uploadState === "uploading" ||
      uploadState === "processing"
    )
      return;
    setUploadState("uploading");
    setUploadProgress(0);
    let p = 0;
    const iv = setInterval(() => {
      p += Math.random() * 14 + 6;
      if (p >= 100) {
        clearInterval(iv);
        setUploadProgress(100);
        setUploadState("processing");
        setTimeout(() => setUploadState("complete"), 2200);
      } else {
        setUploadProgress(p);
      }
    }, 120);
  };

  const isReady = uploadState === "complete";

  return (
    <div
      className="h-screen overflow-hidden flex flex-col bg-background text-foreground"
      style={{ fontFamily: "'Inter', sans-serif" }}
    >
      {/* Ambient glow */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 40% 60% at 10% 50%, rgba(0,100,200,0.05) 0%, transparent 70%), radial-gradient(ellipse 30% 40% at 90% 15%, rgba(0,212,255,0.04) 0%, transparent 60%)",
        }}
      />

      {/* ── Navigation ────────────────────────────────────────────────────── */}
      <nav
        className="h-14 flex-shrink-0 flex items-center px-4 border-b border-border z-50"
        style={{
          background: "rgba(4,8,15,0.95)",
          backdropFilter: "blur(12px)",
        }}
      >
        {/* Logo */}
        <div className="flex items-center gap-2.5 mr-8">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center"
            style={{
              background:
                "linear-gradient(135deg, #00d4ff 0%, #1a6bff 100%)",
              boxShadow: "0 0 16px rgba(0,212,255,0.4)",
            }}
          >
            <Eye className="w-4 h-4 text-white" />
          </div>
          <div>
            <div
              className="text-sm font-bold tracking-[0.2em] text-foreground"
              style={{ fontFamily: "'Orbitron', monospace" }}
            >
              ThermaVision
            </div>
            <div className="text-[8px] text-muted-foreground tracking-[0.18em] leading-none uppercase">
              Bharatiya Antariksh · ISRO 2026
            </div>
          </div>
        </div>

        {/* Nav links */}
        <div className="hidden md:flex items-center gap-0.5 flex-1">
          {[
            "Dashboard",
            "Upload",
            "History",
            "Reports",
            "Settings",
          ].map((item) => (
            <button
              key={item}
              onClick={() => setNavActive(item)}
              className={`px-3 py-1.5 rounded text-sm transition-colors ${
                navActive === item
                  ? "text-cyan-400 bg-cyan-400/10"
                  : "text-muted-foreground hover:text-foreground hover:bg-white/5"
              }`}
            >
              {item}
            </button>
          ))}
        </div>

        {/* Status pill */}
        <div className="hidden md:flex items-center gap-2 mr-4 px-3 py-1.5 rounded-full border border-green-400/25 bg-green-400/8">
          <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
          <span className="text-[10px] text-green-400 font-mono tracking-widest">
            SYSTEM ONLINE
          </span>
        </div>

        {/* User */}
        <div className="flex items-center gap-2">
          <button className="relative p-2 text-muted-foreground hover:text-foreground transition-colors">
            <Bell className="w-4 h-4" />
            <div className="absolute top-1.5 right-1.5 w-1.5 h-1.5 rounded-full bg-cyan-400" />
          </button>
          <div className="flex items-center gap-2 pl-2 border-l border-border">
            <div
              className="w-7 h-7 rounded-full flex items-center justify-center"
              style={{
                background:
                  "linear-gradient(135deg, #00d4ff, #1a6bff)",
              }}
            >
              <User className="w-3.5 h-3.5 text-white" />
            </div>
            <div className="hidden sm:block leading-tight">
              <div className="text-xs font-medium text-foreground">
                Dr. P. Sharma
              </div>
              <div className="text-[10px] text-muted-foreground">
                Senior Analyst · SAC
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* ── Content ───────────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Tab bar */}
        <div
          className="flex-shrink-0 flex items-center border-b border-border px-4"
          style={{ background: "rgba(4,8,20,0.7)" }}
        >
          {[
            {
              id: "dashboard",
              label: "Dashboard",
              icon: Activity,
            },
            {
              id: "heatmap",
              label: "Confidence Heatmap",
              icon: Layers,
            },
            {
              id: "report",
              label: "Analysis Report",
              icon: FileText,
            },
          ].map((t) => {
            const Icon = t.icon;
            return (
              <button
                key={t.id}
                onClick={() => setTab(t.id as typeof tab)}
                className={`flex items-center gap-2 px-4 py-3 text-xs border-b-2 transition-colors ${
                  tab === t.id
                    ? "border-cyan-400 text-cyan-400"
                    : "border-transparent text-muted-foreground hover:text-foreground"
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {t.label}
              </button>
            );
          })}
          <div className="ml-auto flex items-center gap-3 py-2">
            {isReady && (
              <div className="flex items-center gap-1.5 text-[10px] text-green-400 font-mono">
                <CheckCircle className="w-3 h-3" /> Analysis
                complete
              </div>
            )}
            <div className="text-[10px] text-muted-foreground font-mono hidden lg:block">
              Scene: INSAT3D-20260701-0814Z
            </div>
          </div>
        </div>

        {/* Main panels */}
        <div className="flex-1 flex overflow-hidden">
          {/* ── Dashboard tab ── */}
          {tab === "dashboard" && (
            <>
              {/* LEFT PANEL */}
              <div
                className="w-64 xl:w-72 flex-shrink-0 border-r border-border flex flex-col overflow-y-auto gap-3 p-3"
                style={{ background: "rgba(4,8,20,0.6)" }}
              >
                <div className="text-[10px] text-muted-foreground font-mono tracking-widest pt-1">
                  INPUT IMAGERY
                </div>

                {/* Drop zone */}
                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    setIsDragOver(true);
                  }}
                  onDragLeave={() => setIsDragOver(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setIsDragOver(false);
                    simulateUpload();
                  }}
                  className={`relative border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all ${
                    isDragOver
                      ? "border-cyan-400 bg-cyan-400/5"
                      : "border-border hover:border-cyan-400/35"
                  }`}
                  onClick={simulateUpload}
                >
                  <Upload className="w-7 h-7 text-cyan-400 mx-auto mb-2" />
                  <div className="text-xs text-foreground font-medium mb-1">
                    Drop IR Image
                  </div>
                  <div className="text-[10px] text-muted-foreground leading-relaxed">
                    GeoTIFF · HDF5 · NetCDF
                    <br />
                    PNG · ENVI · IMG
                  </div>
                  {isDragOver && (
                    <div className="absolute inset-0 rounded-xl bg-cyan-400/10 flex items-center justify-center">
                      <span className="text-xs text-cyan-400 font-medium">
                        Release to Upload
                      </span>
                    </div>
                  )}
                </div>
                <button
                  onClick={simulateUpload}
                  className="w-full py-2 rounded-lg text-xs font-semibold text-white transition-opacity hover:opacity-90"
                  style={{
                    background:
                      "linear-gradient(90deg, #1a6bff, #00d4ff)",
                  }}
                >
                  Browse Files
                </button>

                {/* Progress */}
                {(uploadState === "uploading" ||
                  uploadState === "processing") && (
                  <div className="bg-card border border-border rounded-xl p-3">
                    <div className="flex justify-between text-[10px] font-mono mb-2">
                      <span className="text-muted-foreground">
                        {uploadState === "uploading"
                          ? "UPLOADING"
                          : "PROCESSING"}
                      </span>
                      <span className="text-cyan-400">
                        {uploadState === "uploading"
                          ? `${Math.round(uploadProgress)}%`
                          : "AI RUNNING"}
                      </span>
                    </div>
                    <div className="h-1.5 bg-[#060e1c] rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-200"
                        style={{
                          width:
                            uploadState === "processing"
                              ? "100%"
                              : `${uploadProgress}%`,
                          background:
                            "linear-gradient(90deg, #1a6bff, #00d4ff)",
                        }}
                      />
                    </div>
                    {uploadState === "processing" && (
                      <div className="mt-2 space-y-1">
                        {[
                          "Registration complete",
                          "Feature extraction done",
                          "RGB synthesis running...",
                        ].map((s, i) => (
                          <div
                            key={s}
                            className="flex items-center gap-1.5 text-[10px]"
                          >
                            {i < 2 ? (
                              <CheckCircle className="w-3 h-3 text-green-400 flex-shrink-0" />
                            ) : (
                              <RefreshCw className="w-3 h-3 text-cyan-400 animate-spin flex-shrink-0" />
                            )}
                            <span className="text-muted-foreground">
                              {s}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Metadata */}
                <div className="bg-card border border-border rounded-xl p-3">
                  <div className="text-[10px] text-muted-foreground font-mono mb-2.5 tracking-widest">
                    SCENE METADATA
                  </div>
                  <div className="space-y-2">
                    {[
                      ["Resolution", "4096 × 3072 px"],
                      ["Sensor", "INSAT-3DR Bnd 5"],
                      ["Timestamp", "01 Jul 2026 08:14"],
                      ["Location", "21.4°N, 78.9°E"],
                      ["Projection", "WGS84 / UTM 44N"],
                      ["File Size", "186.4 MB"],
                    ].map(([k, v]) => (
                      <div
                        key={k}
                        className="flex justify-between gap-1"
                      >
                        <span className="text-[10px] text-muted-foreground">
                          {k}
                        </span>
                        <span className="text-[10px] text-foreground font-mono text-right">
                          {v}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Pipeline status */}
                <div className="bg-card border border-border rounded-xl p-3">
                  <div className="text-[10px] text-muted-foreground font-mono mb-2.5 tracking-widest">
                    PIPELINE STATUS
                  </div>
                  <div className="space-y-1.5">
                    {[
                      "Registration",
                      "SSL Pre-training",
                      "Feature Extraction",
                      "Image Enhancement",
                      "RGB Translation",
                      "Object Detection",
                      "Confidence Estimation",
                    ].map((step) => (
                      <div
                        key={step}
                        className="flex items-center gap-2"
                      >
                        <CheckCircle className="w-3 h-3 text-green-400 flex-shrink-0" />
                        <span className="text-[10px] text-muted-foreground">
                          {step}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* CENTER PANEL */}
              <div className="flex-1 flex flex-col overflow-hidden">
                {/* Toolbar */}
                <div
                  className="flex-shrink-0 flex items-center gap-1.5 px-3 py-2 border-b border-border"
                  style={{ background: "rgba(4,8,20,0.4)" }}
                >
                  <span className="text-[10px] text-muted-foreground font-mono mr-1.5 tracking-wider">
                    IMAGE VIEWER
                  </span>
                  {[ZoomIn, ZoomOut, Maximize2].map(
                    (Icon, i) => (
                      <button
                        key={i}
                        className="p-1.5 text-muted-foreground hover:text-foreground rounded transition-colors hover:bg-white/5"
                      >
                        <Icon className="w-3.5 h-3.5" />
                      </button>
                    ),
                  )}
                  <div className="ml-auto flex items-center gap-2 text-[10px] text-muted-foreground font-mono">
                    <span className="hidden sm:block">
                      ← Drag slider to compare IR vs RGB →
                    </span>
                  </div>
                </div>

                {/* Comparison slider */}
                <div className="flex-1 p-3 overflow-hidden">
                  <ImageSlider />
                </div>
              </div>

              {/* RIGHT PANEL */}
              <div
                className="w-72 xl:w-80 flex-shrink-0 border-l border-border flex flex-col overflow-y-auto gap-3 p-3"
                style={{ background: "rgba(4,8,20,0.6)" }}
              >
                <div className="flex items-center justify-between pt-1">
                  <div className="text-[10px] text-muted-foreground font-mono tracking-widest">
                    AI ANALYSIS
                  </div>
                  <div className="flex items-center gap-1.5 text-[10px] text-green-400 font-mono">
                    <div className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                    COMPLETE
                  </div>
                </div>

                {/* Detected objects */}
                <div className="bg-card border border-border rounded-xl p-3">
                  <div className="text-[10px] text-muted-foreground font-mono mb-3 tracking-widest">
                    DETECTED OBJECTS
                  </div>
                  <div className="space-y-3">
                    {DETECTED_OBJECTS.map((obj) => (
                      <div key={obj.id}>
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs text-foreground">
                            {obj.label}
                          </span>
                          <span
                            className="text-xs font-mono"
                            style={{ color: obj.color }}
                          >
                            {obj.confidence}%
                          </span>
                        </div>
                        <div className="h-1 bg-[#060e1c] rounded-full overflow-hidden mb-0.5">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${obj.confidence}%`,
                              backgroundColor: obj.color,
                            }}
                          />
                        </div>
                        <div className="text-[10px] text-muted-foreground">
                          {obj.count}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Radar */}
                <div className="bg-card border border-border rounded-xl p-3">
                  <div className="text-[10px] text-muted-foreground font-mono mb-1 tracking-widest">
                    DETECTION CONFIDENCE
                  </div>
                  <ResponsiveContainer
                    width="100%"
                    height={150}
                  >
                    <RadarChart
                      data={RADAR_DATA}
                      margin={{
                        top: 8,
                        right: 12,
                        bottom: 8,
                        left: 12,
                      }}
                    >
                      <PolarGrid stroke="rgba(0,212,255,0.1)" />
                      <PolarAngleAxis
                        dataKey="subject"
                        tick={{ fill: "#4a7a9a", fontSize: 9 }}
                      />
                      <Radar
                        dataKey="value"
                        stroke="#00d4ff"
                        fill="#00d4ff"
                        fillOpacity={0.15}
                        strokeWidth={1.5}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>

                {/* Confidence scores */}
                <div className="bg-card border border-border rounded-xl p-3">
                  <div className="text-[10px] text-muted-foreground font-mono mb-3 tracking-widest">
                    CONFIDENCE SCORES
                  </div>
                  <div className="space-y-2.5">
                    {[
                      {
                        label: "Overall Reliability",
                        value: "91.2%",
                        color: "text-green-400",
                      },
                      {
                        label: "Prediction Accuracy",
                        value: "88.7%",
                        color: "text-cyan-400",
                      },
                      {
                        label: "Image Quality Index",
                        value: "94.1",
                        color: "text-blue-400",
                      },
                    ].map((m) => (
                      <div
                        key={m.label}
                        className="flex items-center justify-between"
                      >
                        <span className="text-xs text-muted-foreground">
                          {m.label}
                        </span>
                        <span
                          className={`text-sm font-bold font-mono ${m.color}`}
                        >
                          {m.value}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Actions */}
                <div className="space-y-1.5">
                  {[
                    {
                      label: "View Heatmap",
                      icon: Layers,
                      onClick: () => setTab("heatmap"),
                      iconColor: "text-cyan-400",
                    },
                    {
                      label: "Generate Report",
                      icon: FileText,
                      onClick: () => setTab("report"),
                      iconColor: "text-cyan-400",
                    },
                    {
                      label: "Export Results",
                      icon: Download,
                      onClick: () => {},
                      iconColor: "text-green-400",
                    },
                  ].map((a) => {
                    const Icon = a.icon;
                    return (
                      <button
                        key={a.label}
                        onClick={a.onClick}
                        className="w-full flex items-center justify-between px-3 py-2.5 bg-card border border-border rounded-lg text-xs text-foreground hover:border-cyan-400/25 transition-colors"
                      >
                        <span className="flex items-center gap-2">
                          <Icon
                            className={`w-3.5 h-3.5 ${a.iconColor}`}
                          />{" "}
                          {a.label}
                        </span>
                        <ChevronRight className="w-3.5 h-3.5 text-muted-foreground" />
                      </button>
                    );
                  })}
                </div>
              </div>
            </>
          )}

          {/* ── Heatmap tab ── */}
          {tab === "heatmap" && (
            <div className="flex-1 overflow-hidden">
              <HeatmapView />
            </div>
          )}

          {/* ── Report tab ── */}
          {tab === "report" && (
            <div className="flex-1 overflow-y-auto">
              <ReportView />
            </div>
          )}
        </div>

        {/* ── Stats bar ─────────────────────────────────────────────────── */}
        <div
          className="flex-shrink-0 border-t border-border flex items-center gap-1 px-4 py-2.5 overflow-x-auto"
          style={{ background: "rgba(4,8,20,0.85)" }}
        >
          {STATS.map((s, i) => {
            const Icon = s.icon;
            return (
              <div
                key={s.label}
                className="flex items-center gap-2 flex-shrink-0"
              >
                <div className="w-6 h-6 rounded-md bg-card border border-border flex items-center justify-center">
                  <Icon className="w-3 h-3 text-cyan-400" />
                </div>
                <div>
                  <div className="text-[9px] text-muted-foreground font-mono leading-none">
                    {s.label}
                  </div>
                  <div className="text-xs font-bold font-mono text-foreground leading-none mt-0.5">
                    {s.value}
                    {s.unit && (
                      <span className="text-[9px] font-normal text-muted-foreground ml-0.5">
                        {s.unit}
                      </span>
                    )}
                  </div>
                </div>
                {i < STATS.length - 1 && (
                  <div className="w-px h-5 bg-border ml-1.5" />
                )}
              </div>
            );
          })}
          <div className="ml-auto text-[9px] text-muted-foreground font-mono hidden lg:block flex-shrink-0">
            ISRO Bharatiya Antariksh Hackathon 2026 · Team IRIS
          </div>
        </div>

        {/* ── AI Pipeline ────────────────────────────────────────────────── */}
        <div
          className="flex-shrink-0 border-t border-border px-4 py-3"
          style={{ background: "rgba(4,6,14,0.95)" }}
        >
          <div className="flex items-center justify-between mb-2.5">
            <div className="text-[9px] text-muted-foreground font-mono tracking-[0.2em]">
              AI PROCESSING PIPELINE
            </div>
            <div className="text-[9px] text-cyan-400/70 font-mono hidden sm:block">
              PyTorch 2.3 · CUDA 12.2 · Model: IRIS-v2.4.1-SAC
            </div>
          </div>
          <div className="flex items-start gap-0 overflow-x-auto pb-1">
            {PIPELINE_STEPS.map((step, i) => {
              const Icon = step.icon;
              return (
                <div
                  key={step.id}
                  className="flex items-center flex-shrink-0"
                >
                  <div className="flex flex-col items-center">
                    <div
                      className="w-14 h-14 rounded-xl flex flex-col items-center justify-center border transition-all hover:scale-105 cursor-default"
                      style={{
                        background: `${step.color}10`,
                        borderColor: `${step.color}28`,
                        boxShadow: `0 0 10px ${step.color}08`,
                      }}
                    >
                      <Icon
                        className="w-4 h-4 mb-0.5"
                        style={{ color: step.color }}
                      />
                      <div
                        className="text-[7px] font-mono"
                        style={{ color: `${step.color}90` }}
                      >
                        {String(step.id).padStart(2, "0")}
                      </div>
                    </div>
                    <div className="text-[9px] text-center text-muted-foreground mt-1 max-w-[60px] leading-tight">
                      {step.name}
                    </div>
                  </div>
                  {i < PIPELINE_STEPS.length - 1 && (
                    <div className="flex items-center mb-5 mx-0.5 flex-shrink-0">
                      <div
                        className="w-4 h-px"
                        style={{
                          background:
                            "linear-gradient(90deg, rgba(0,212,255,0.3), rgba(0,212,255,0.08))",
                        }}
                      />
                      <ArrowRight
                        className="w-2.5 h-2.5"
                        style={{ color: "rgba(0,212,255,0.3)" }}
                      />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* ── Tech badges ────────────────────────────────────────────────── */}
        <div
          className="flex-shrink-0 border-t border-border px-4 py-2 flex items-center gap-1.5 flex-wrap"
          style={{ background: "rgba(3,5,10,0.98)" }}
        >
          <span className="text-[9px] text-muted-foreground font-mono tracking-widest mr-1">
            STACK
          </span>
          {TECH_BADGES.map((t) => (
            <span
              key={t.name}
              className="px-2 py-0.5 rounded text-[9px] font-mono border"
              style={{
                color: t.color,
                borderColor: `${t.color}28`,
                background: `${t.color}0c`,
              }}
            >
              {t.name}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}