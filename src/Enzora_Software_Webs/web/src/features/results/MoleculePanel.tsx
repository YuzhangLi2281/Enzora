import type { Candidate, HostProfile } from '../../shared/api/contracts';
import { t } from '../../shared/lib/i18n';
import { useApp } from '../../app/store';
import { SequenceCompare } from './SequenceCompare';

export function MoleculePanel(props: {
  wtSequence: string;
  candidates: Candidate[];
  codonRows: HostProfile[];
}) {
  const lang = useApp((s) => s.lang);
  const selectedMutation = useApp((s) => s.selectedMutation);
  const setSelectedMutation = useApp((s) => s.setSelectedMutation);
  return (
    <div className="sequence-panel">
      <div style={{ flex: 1, minHeight: 0, overflow: 'auto' }}>
        <SequenceCompare {...props} selectedMutation={selectedMutation} onPick={setSelectedMutation} />
      </div>
      <div className="small dim" style={{ marginTop: 10 }}>{t('molecule.disclaimer', lang)}</div>
    </div>
  );
}
