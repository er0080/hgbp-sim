import { useState } from "react";
import { api, hms } from "./api";
import { useStand } from "./useStand";
import OperatorPanel from "./components/OperatorPanel";
import TrendsTab from "./components/TrendsTab";
import SettingsTab from "./components/SettingsTab";
import InitDialog from "./components/InitDialog";
import AlarmBar from "./components/AlarmBar";

type Tab = "operator" | "trends" | "settings";

export default function App() {
  const { snap, connected, history, histVersion } = useStand();
  const [tab, setTab] = useState<Tab>("operator");
  const [init, setInit] = useState<null | "cold" | "warm">(null);

  const setSpeed = (v: number) => api("/api/sim", { speed_factor: v });
  const togglePause = () => api("/api/sim", { paused: !snap?.paused });

  return (
    <div className="app">
      <div className="topbar">
        <h1>HGBP Test Stand</h1>
        <span className="dot" data-on={connected} style={{ background: connected ? "var(--ok)" : "var(--bad)" }} />
        <span className="note">{connected ? "live" : "connecting"}</span>
        <div className="tabs">
          {(["operator", "trends", "settings"] as Tab[]).map((t) => (
            <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
              {t === "operator" ? "Operator" : t === "trends" ? "Trends" : "Settings"}
            </button>
          ))}
        </div>
        <div className="spacer" />
        {snap && (
          <>
            <span className="note">{snap.fluid}</span>
            <span className="clock">{hms(snap.t)}</span>
            <label className="note">speed</label>
            <select value={snap.speed_factor} onChange={(e) => setSpeed(Number(e.target.value))}>
              {[0.5, 1, 2, 5, 10, 20, 50].map((v) => (
                <option key={v} value={v}>{v}x</option>
              ))}
            </select>
            <button onClick={togglePause} className={snap.paused ? "ok" : ""}>{snap.paused ? "Resume" : "Pause"}</button>
            <button onClick={() => setInit("cold")}>Cold start</button>
            <button onClick={() => setInit("warm")}>Warm start</button>
          </>
        )}
      </div>
      {snap && <AlarmBar snap={snap} />}
      <div className="main">
        {!snap && <p className="note">Waiting for the simulator...</p>}
        {snap && tab === "operator" && <OperatorPanel snap={snap} />}
        {snap && tab === "trends" && <TrendsTab history={history} version={histVersion} snap={snap} />}
        {snap && tab === "settings" && <SettingsTab snap={snap} />}
      </div>
      {init && snap && <InitDialog mode={init} snap={snap} onClose={() => setInit(null)} />}
    </div>
  );
}
