import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';

const source = fs.readFileSync(new URL('../src/features/experiment/review.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }).outputText;
const module = { exports: {} };
vm.runInNewContext(`(function(exports) { ${compiled}\n})`, {})(module.exports);
const { automaticReview, candidateError, defaultCandidate, defaultSelection, positionList, resolvedReview, selectionError, sequenceError, reviewedRequest, conversationUpdate } = module.exports;

test('neutral defaults activate only requested objectives; API weights remain bounded', () => {
  const selection = defaultSelection();
  assert.equal(JSON.stringify(selection.weights), JSON.stringify({ph: 0, activity: .5, stability: .5}));
  assert.equal(selectionError(selection), null);
  const target = defaultSelection(7.5);
  assert.equal(target.weights.ph, 1/3);
  assert.equal(selectionError(target), null);
  for (const weights of [{ph:1.1,activity:0,stability:0},{ph:.3,activity:.6,stability:.3},{ph:0,activity:0,stability:0},{ph:-.1,activity:.8,stability:.3}])
    assert.equal(selectionError({...selection,weights}), 'weights');
});

test('AI review preserves position, activity, stability and pool constraints on confirmed submission', () => {
  const parsed = { resolved_candidate: { top_k: 24, allowed_positions: [3, 8], excluded_positions: [5] }, resolved_selection: { target_ph: 7.5, min_activity_log10: 0.25, stability_direction: 'higher_is_better', stability_operator: 'ge', stability_threshold: 1, top_k: 8, weights: { ph: 0.3, activity: 0.4, stability: 0.3 } } };
  const draft = resolvedReview(parsed, defaultCandidate(), defaultSelection());
  const request = reviewedRequest({ ...draft, name: 'sample', sequence: 'MKTAY', goal: 'pH 7.5', aiReady: true, lang: 'zh' });
  assert.equal(request.candidate.top_k, 24);
  assert.deepEqual(Array.from(request.candidate.allowed_positions), [3, 8]);
  assert.deepEqual(Array.from(request.candidate.excluded_positions), [5]);
  assert.equal(request.selection.min_activity_log10, 0.25);
  assert.equal(request.selection.stability_operator, 'ge');
  assert.equal(request.selection.stability_threshold, 1);
  assert.equal(request.selection.weights.ph, 0.3);
  assert.equal(request.llm.parameter_parsing, false);
  assert.equal(request.llm.result_analysis, true);
  assert.equal(request.llm.enabled, true);
  assert.deepEqual(Object.keys(request.llm).sort(), ["analysis_fail_open", "enabled", "parameter_parsing", "result_analysis"]);
  assert.equal(draft.selection.weights.ph, 0.3);
});

test('protein validation does not turn prose or unknown residues into sequences', () => {
  assert.equal(sequenceError('MKTAYIAK'), null);
  assert.equal(sequenceError('Please optimize this protein'), 'alphabet');
  assert.equal(sequenceError('MKT X'), 'alphabet');
  assert.equal(sequenceError('M'.repeat(1023)), 'length');
  assert.equal(sequenceError(''), 'empty');
});

test('conversational goal updates preserve the chosen sequence and append the exact original request', () => {
  const addition = 'Also exclude residues 3 and 8. Keep pH at 7.5.';
  const update = conversationUpdate({ action: 'update_goal', request_text: addition, records: [] }, true, 'Prefer higher activity.');
  assert.equal(update.preserveDraft, true);
  assert.equal(update.goal, `Prefer higher activity.\n${addition}`);
  assert.equal(conversationUpdate({ action: 'update_goal', request_text: addition }, false, '').preserveDraft, false);
});

test('AI clarification leaves the existing goal intact and a new source replaces the draft', () => {
  const clarification = conversationUpdate({ action: 'clarify', records: [] }, true, 'target pH 8');
  assert.equal(clarification.preserveDraft, true);
  assert.equal(clarification.goal, null);
  assert.equal(conversationUpdate({ action: 'new_input', records: [{ id: 'new', name: 'new source', type: 'protein', sequence: 'MKTA' }] }, true, 'target pH 8').preserveDraft, false);
});

test('prepared sequences automatically obtain editable AI values using the environment credential without a session', async () => {
  const calls = [];
  const preview = { resolved_candidate: { top_k: 32, allowed_positions: null, excluded_positions: [3] }, resolved_selection: { target_ph: 7.5, top_k: 10, weights: { ph: 0.5, activity: 0.3, stability: 0.2 } }, suggested_fields: ['selection.weights.ph'], explicit_fields: ['selection.target_ph'], rationale: { 'selection.weights.ph': 'The target pH is the primary goal.' }, source: 'ai' };
  const review = await automaticReview({ name: 'CAII', sequence: 'MKTAYIAK', goal: 'Improve activity near pH 7.5.', candidate: defaultCandidate(), selection: defaultSelection(), sessionId: null, aiReady: true, lang: 'en', parse: async (...args) => { calls.push(args); return preview; } });
  assert.equal(calls.length, 1);
  assert.equal(calls[0][0], 'Improve activity near pH 7.5.');
  assert.equal(calls[0][3], null);
  assert.equal(calls[0][4].sequence_name, 'CAII');
  assert.equal(calls[0][4].sequence_type, 'protein');
  assert.equal(calls[0][4].sequence_length, 8);
  assert.equal(review.selection.target_ph, 7.5);
  assert.equal(review.selection.weights.ph, 0.5);
  assert.equal(review.parsed, preview);

  // A user adjustment after generation must survive submission; the workflow
  // must not run a second parameter parse and replace this reviewed form.
  const changed = { ...review.selection, top_k: 5, weights: { ph: 1, activity: 2, stability: 1 } };
  const request = reviewedRequest({ name: 'CAII', sequence: 'MKTAYIAK', goal: 'Improve activity near pH 7.5.', candidate: { ...review.candidate, top_k: 24 }, selection: changed, aiReady: true, lang: 'en' });
  assert.equal(request.selection.top_k, 5);
  assert.equal(request.candidate.top_k, 24);
  assert.equal(request.selection.weights.activity, 0.5);
  assert.equal(request.llm.parameter_parsing, false);
  assert.equal(calls.length, 1);
});

test('sequence-only input requests labeled initial suggestions rather than pretending the user set a pH', async () => {
  let sentGoal = '';
  const review = await automaticReview({ name: 'sample', sequence: 'MKTAY', goal: '', candidate: defaultCandidate(), selection: defaultSelection(), sessionId: null, aiReady: true, lang: 'zh', parse: async (goal) => { sentGoal = goal; return { resolved_candidate: {}, resolved_selection: { weights: { ph: 0, activity: 0.5, stability: 0.5 } } }; } });
  assert.match(sentGoal, /初始酶设计参数/);
  assert.match(sentGoal, /不要把建议描述成用户指定条件/);
  assert.equal(review.selection.target_ph, null);
  assert.equal(review.selection.weights.activity, 0.5);
});

test('automatic parameter generation waits for a valid prepared protein and a configured AI', async () => {
  let calls = 0;
  const input = { name: 'sample', sequence: 'MKTAY', goal: 'More stable', candidate: defaultCandidate(), selection: defaultSelection(), sessionId: null, aiReady: true, lang: 'en', parse: async () => { calls++; return { resolved_candidate: {}, resolved_selection: {} }; } };
  assert.equal(await automaticReview({ ...input, aiReady: false }), null);
  assert.equal(await automaticReview({ ...input, sequence: '' }), null);
  assert.equal(await automaticReview({ ...input, sequence: 'Please optimize activity' }), null);
  assert.equal(calls, 0);
});

test('edited pool and position fields reject invalid or contradictory candidate constraints', () => {
  const selection = defaultSelection();
  const candidate = defaultCandidate();
  assert.equal(candidateError(candidate, selection, 50), null);
  assert.equal(candidateError({ ...candidate, top_k: 0 }, selection, 50), 'candidate_pool');
  assert.equal(candidateError({ ...candidate, top_k: 4 }, selection, 50), 'pool_too_small');
  assert.equal(candidateError({ ...candidate, allowed_positions: [51] }, selection, 50), 'positions');
  assert.equal(candidateError({ ...candidate, excluded_positions: [0] }, selection, 50), 'positions');
  assert.equal(candidateError({ ...candidate, allowed_positions: [3], excluded_positions: [3] }, selection, 50), 'position_conflict');
  assert.deepEqual(Array.from(positionList('3, 8，25; 8')), [3, 8, 25]);
  assert.equal(candidateError({ ...candidate, allowed_positions: positionList('3, bad') }, selection, 50), 'positions');
  assert.equal(selectionError({ ...selection, stability_operator: 'le', stability_threshold: null }), 'stability_required');
});


test('library request defaults to 1000 singles and carries explicit substrate and editable mutation order', () => {
  assert.equal(defaultCandidate().top_k, 1000);
  const request = reviewedRequest({ name: 'enzyme', sequence: 'M'.repeat(60), goal: '', candidate: defaultCandidate(), selection: defaultSelection(), aiReady: false, lang: 'en', ligandSmiles: 'CCO', mutationOrder: 2 });
  assert.equal(request.mutation.counts[2], 1000);
  assert.equal(request.activity.ligand_smiles, 'CCO');
  assert.equal(request.activity.task, 'kcat');
  assert.equal(request.runtime.phase_control, true);
  assert.equal(request.selection.confidence.activity, 1);
});
