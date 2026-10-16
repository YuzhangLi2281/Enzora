/** Panel 4: weight donut (live re-ranking) + candidate table. */

import { useEffect, useMemo, useState } from 'react';
import { rerankCandidates } from './rerank';
import { t } from '../../shared/lib/i18n';
import { useApp } from '../../app/store';
import type { Candidate } from '../../shared/api/contracts';
import { CandidateTable } from './CandidateTable';
import { WeightDonut } from './WeightDonut';

export function CandidatePanel(props: { raw: Candidate[]; wtSequence: string }) {
  const lang = useApp((s) => s.lang);
  const weights = useApp((s) => s.weights);
  const weightsDirty = useApp((s) => s.weightsDirty);
  const setWeights = useApp((s) => s.setWeights);
  const resetWeights = useApp((s) => s.resetWeights);
  const runRequest = useApp((s) => s.runRequest);
  const [adjust, setAdjust] = useState(false);

  const ranked = useMemo(
    () => !weightsDirty ? props.raw : rerankCandidates(props.raw, weights, {
      confidence: runRequest?.selection.confidence,
      target_ph: runRequest?.selection.target_ph ?? null,
      ph_tolerance: runRequest?.selection.ph_tolerance ?? null,
      stability_direction: runRequest?.selection.stability_direction ?? 'lower_is_better',
    }),
    [props.raw, weights, runRequest, weightsDirty],
  );

  useEffect(() => {
    // keep the store's candidate list in sync for cross-panel linkage
    useApp.getState().setCandidates(ranked, props.raw);
  }, [ranked, props.raw]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
      <div style={{ marginBottom: 10 }}>
        <WeightDonut
          size={60}
          weights={weights}
          applied={!adjust && !weightsDirty && Boolean(props.raw.length)}
          adjust={adjust}
          onChange={(w) => setWeights(w, true)}
        />
        <div className="chip-row" style={{ marginTop: 8 }}>
          <button className={`chip ${adjust ? 'active' : ''}`} onClick={() => setAdjust(!adjust)}>
            {t('weights.adjust', lang)}
          </button>
          {weightsDirty && (
            <button className="chip" onClick={resetWeights}>{t('weights.reset', lang)}</button>
          )}
        </div>
      </div>
      <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
        <CandidateTable rows={ranked} />
      </div>
    </div>
  );
}
