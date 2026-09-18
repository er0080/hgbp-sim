import { useEffect, useRef } from "react";
import uPlot from "uplot";
import type { History } from "../types";

export interface SeriesDef { key: string; label: string; color: string; dash?: number[]; width?: number }

function fmtTime(v: number) {
  const s = Math.max(0, Math.floor(v));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(ss).padStart(2, "0")}` : `${m}:${String(ss).padStart(2, "0")}`;
}

export default function UPlotChart({ title, series, history, version, windowS, unit }:
  { title: string; series: SeriesDef[]; history: React.MutableRefObject<History>; version: number; windowS: number; unit?: string }) {
  const el = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);

  const buildData = (): uPlot.AlignedData => {
    const h = history.current;
    const t = h.t ?? [];
    const n = t.length;
    let i0 = 0;
    if (n && windowS > 0) {
      const tmin = t[n - 1] - windowS;
      i0 = t.findIndex((v) => v >= tmin);
      if (i0 < 0) i0 = 0;
    }
    const xs = t.slice(i0);
    const ys = series.map((s) => (h[s.key] ?? []).slice(i0).map((v) => (Number.isFinite(v) ? v : null)));
    return [xs, ...ys] as uPlot.AlignedData;
  };

  useEffect(() => {
    if (!el.current) return;
    const opts: uPlot.Options = {
      title,
      width: el.current.clientWidth,
      height: 240,
      scales: { x: { time: false } },
      axes: [
        { stroke: "#8b98a5", grid: { stroke: "#243040" }, ticks: { stroke: "#243040" }, values: (_u, vals) => vals.map(fmtTime) },
        { stroke: "#8b98a5", grid: { stroke: "#243040" }, ticks: { stroke: "#243040" }, label: unit, labelSize: 14 },
      ],
      series: [
        { label: "t", value: (_u, v) => (v == null ? "" : fmtTime(v)) },
        ...series.map((s) => ({ label: s.label, stroke: s.color, width: s.width ?? 1.5, dash: s.dash, value: (_u: uPlot, v: number | null) => (v == null ? "--" : v.toFixed(2)) })),
      ],
      legend: { show: true },
      cursor: { sync: { key: "hgbp" } },
    };
    plot.current = new uPlot(opts, buildData(), el.current);
    const ro = new ResizeObserver(() => {
      if (el.current && plot.current) plot.current.setSize({ width: el.current.clientWidth, height: 240 });
    });
    ro.observe(el.current);
    return () => { ro.disconnect(); plot.current?.destroy(); plot.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [title, series.map((s) => s.key).join(",")]);

  useEffect(() => {
    plot.current?.setData(buildData());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version, windowS]);

  return <div className="chart" ref={el} />;
}
