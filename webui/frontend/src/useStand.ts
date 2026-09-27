import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { History, Snapshot } from "./types";

const MAX_POINTS = 60000;
const TRIM_SLACK = 5000;       // trim the buffer in chunks, not on every message

export function useStand() {
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [connected, setConnected] = useState(false);
  const history = useRef<History>({ t: [] });
  const [histVersion, setHistVersion] = useState(0);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let closed = false;
    let timer: number | undefined;
    const connect = async () => {
      try {
        const h = await api<History>("/api/history?max_points=15000");
        history.current = h;
        setHistVersion((v) => v + 1);
      } catch { /* backend not ready yet */ }
      // the stream continues after the last row we hold (no gap, no overlap)
      const tHeld = history.current.t ?? [];
      const since = tHeld.length ? `?since=${tHeld[tHeld.length - 1]}` : "";
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws${since}`);
      ws.onopen = () => setConnected(true);
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type !== "step") return;
        setSnap(msg.data as Snapshot);
        const rows = msg.rows as Record<string, number[]> | undefined;
        if (rows && rows.t && rows.t.length) {
          const h = history.current;
          const tArr = h.t;
          if (tArr.length && rows.t[0] < tArr[tArr.length - 1]) {
            for (const k of Object.keys(h)) h[k] = [];       // backend history restarted
          }
          for (const k of Object.keys(rows)) {
            if (!h[k]) h[k] = [];
            const arr = h[k];
            for (const v of rows[k]) arr.push(v);
            if (arr.length > MAX_POINTS + TRIM_SLACK) arr.splice(0, arr.length - MAX_POINTS);
          }
          setHistVersion((v) => v + 1);
        }
      };
      ws.onclose = () => {
        setConnected(false);
        if (!closed) timer = window.setTimeout(connect, 1000);
      };
      ws.onerror = () => ws?.close();
    };
    connect();
    return () => {
      closed = true;
      if (timer) clearTimeout(timer);
      ws?.close();
    };
  }, []);
  return { snap, connected, history, histVersion };
}
