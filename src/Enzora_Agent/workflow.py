from __future__ import annotations

import copy
import csv
import json
import traceback
import time
import yaml
from pathlib import Path

from .adapters import EpHodAdapter, PhenomAdapter, UniStabAdapter
from .config import validate_candidate_and_selection
from .outputs.codon_optimization import resolve_codon_optimization_request, write_codon_optimization_outputs
from .core.runtime import build_env, create_try_dir, write_candidates_csv, write_json, cleanup_run_cache
from .outputs.report import build_markdown_report
from .core.screening import annotate_candidates, mark_pareto_optimal, write_mutation_features_csv
from .adapters.vesm import VESMAdapter
from .core.screening import rank_candidates, balance_weights
from .core.types import Candidate, BioArtifact
from .outputs.visualization import generate_visualizations


STAGES = [
    "resolve_user_request",
    "vesm_library_construction",
    "phenom_activity_prediction",
    "ephod_ph_prediction",
    "unistab_stability_prediction",
    "personalized_filter_and_rank",
    "point_mutation_feature_analysis",
    "codon_optimization",
    "visualize_results",
    "llm_result_analysis",
    "write_markdown_report",
]




class FixedEnzymeWorkflow:
    def __init__(
        self,
        cfg: dict,
        *,
        force_cpu: bool = False,
        api_key: str | None = None,
        print_llm: bool = False,
    ):
        self.cfg = cfg
        self.force_cpu = force_cpu
        self.api_key = api_key
        self.print_llm = print_llm
        m = cfg["models"]
        self.ephod = EpHodAdapter(m["ephod"]["repo"], m["ephod"]["python"], m["ephod"]["weights_dir"])
        self.phenom = PhenomAdapter(m['phenom'])
        self.vesm = VESMAdapter(cfg['mutation'])
        self.unistab = UniStabAdapter(
            m["unistab"]["repo"],
            m["unistab"]["python"],
            m["unistab"]["checkpoint"], m["unistab"]["weights_dir"],
        )

    def _path_preflight(self) -> None:
        errors: list[str] = []
        for adapter in [self.vesm, self.ephod, self.phenom, self.unistab]:
            errors.extend(adapter.preflight())
        if errors:
            raise RuntimeError("模型/路径预检失败:\n- " + "\n- ".join(errors))

    def _llm_client(self, cfg: dict):
        from .llm.gemini import create_client
        return create_client(cfg['llm'], self.api_key)

    def _phase(self, run_dir: Path, phase: int, cfg: dict) -> None:
        if phase > 1 and cfg['runtime']['phase_control']:
            write_json(run_dir / 'progress.json', {'phase': phase - 1, 'status': 'awaiting_next', 'next_phase': phase})
            deadline = time.monotonic() + cfg['runtime']['timeout_seconds']
            while not (run_dir / f'advance_{phase}').is_file():
                if time.monotonic() > deadline:
                    raise TimeoutError(f'Awaiting Next Process for phase {phase} timed out')
                time.sleep(0.25)
        write_json(run_dir / 'progress.json', {'phase': phase, 'status': 'running'})

    def run(self) -> dict:
        cfg = copy.deepcopy(self.cfg)
        base_out_dir = Path(cfg["output_dir"])
        run_dir = create_try_dir(base_out_dir)
        report_language = str(cfg.get("report", {}).get("language", "zh"))
        report: dict = {
            "stages": [],
            "force_cpu": self.force_cpu,
            "base_output_dir": str(base_out_dir),
            "run_dir": str(run_dir),
            "report_language": report_language,
            "status": "running",
        }

        try:
            parsed_request: dict = {}
            llm_cfg = cfg.get("llm", {})
            if bool(llm_cfg.get("enabled", False)) and bool(llm_cfg.get("parameter_parsing", True)):
                from .llm.parameter_parser import parse_user_request

                client = self._llm_client(cfg)
                cfg, parsed_request = parse_user_request(cfg, client=client)
                if self.print_llm:
                    print("\n[LLM 参数解析]")
                    print(json.dumps(parsed_request, ensure_ascii=False, indent=2))
                report["stages"].append({
                    "stage": STAGES[0],
                    "enabled": True,
                    "interpretation": parsed_request.get("interpretation"),
                })
            else:
                validate_candidate_and_selection(cfg, strict_ph_target=True)
                report["stages"].append({"stage": STAGES[0], "enabled": False})

            # Codon optimization is triggered deterministically from the original user request.
            # It is intentionally independent of the LLM so LLM cannot invent a host species.
            codon_request = resolve_codon_optimization_request(cfg.get("user_request"))

            write_json(
                run_dir / "00_resolved_request.json",
                {
                    "sequence": cfg["sequence"],
                    "mutation": cfg["mutation"],
                    "activity": cfg["activity"],
                    "user_request": cfg.get("user_request"),
                    "llm_enabled": bool(llm_cfg.get("enabled", False)),
                    "llm_parsed": parsed_request,
                    "resolved_candidate": cfg["candidate"],
                    "resolved_selection": cfg["selection"],
                    "codon_optimization": codon_request,
                    "report_language": report_language,
                },
            )

            self._path_preflight()
            seq = cfg["sequence"]
            runtime = cfg["runtime"]
            env = build_env(
                force_cpu=self.force_cpu,
                gpu_id=int(runtime["gpu_id"]),
                cache_dir=runtime["cache_dir"],
            )
            temp_dir = run_dir / 'tmp'
            temp_dir.mkdir()
            env['TMPDIR'] = str(temp_dir)
            timeout = int(runtime["timeout_seconds"])

            self._phase(run_dir, 1, cfg)
            # Generation scores describe library quality only; PHENOM alone predicts activity.
            variants = self.vesm.build(seq, cfg['candidate'], work_dir=run_dir / 'vesm',
                                       env=env, timeout=timeout, force_cpu=self.force_cpu)
            candidates = [Candidate(mutation=row['mutations'], wt_sequence=seq,
                                    mutant_sequence=row['sequence'],
                                    library_score=float(row['ensemble_llr_sum'])) for row in variants]
            with (run_dir / '01_mutation_library.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(variants[0]))
                writer.writeheader(); writer.writerows(variants)
            report['stages'].append({'stage': STAGES[1], 'count': len(candidates)})
            write_json(run_dir / 'progress.json', {'phase': 1, 'saved_sequences': len(candidates),
                                                  'target_sequences': cfg['candidate']['top_k']})
            artifact = BioArtifact(data={'protein_sequence': seq,
                'protein_sequences': {c.mutation: c.mutant_sequence for c in candidates},
                'ligand_smiles': cfg['activity']['ligand_smiles']},
                metadata={'sequence_name': cfg['sequence_name'], 'user_request': cfg.get('user_request'),
                          'source': 'VESM library', 'run_name': run_dir.name})
            self._phase(run_dir, 2, cfg)
            activity = self.phenom.predict(
                {'WT': seq, **artifact['protein_sequences']},
                ligand_smiles=artifact['ligand_smiles'], work_dir=run_dir / 'phenom',
                env=env, timeout=timeout, force_cpu=self.force_cpu)
            for candidate in candidates:
                values = activity[candidate.mutation]
                candidate.activity_log10 = values['mu_log10']
                candidate.activity_sigma_log10 = values['sigma_log10']
                candidate.activity_kcat = values['prediction']
            annotate_candidates(candidates)
            write_candidates_csv(run_dir / 'scorer_predictions.csv', candidates)
            report['stages'].append({'stage': STAGES[2], 'count': len(candidates),
                                     'task': 'kcat', 'score_unit': 'log10(s^-1)'})

            # 3) EpHod: WT + pool mutants in one FASTA/model load.
            ph_input = {"WT": seq}
            ph_input.update({
                c.mutation: c.mutant_sequence
                for c in candidates
                if c.mutant_sequence
            })
            ph_details = self.ephod.predict_with_uncertainty(
                ph_input,
                work_dir=run_dir / "ephod",
                env=env,
                timeout=timeout,
            )
            artifact['activity_predictions'] = activity
            ph = {name: value['score'] for name, value in ph_details.items()}
            artifact['ph_predictions'] = ph_details
            wt_ph = ph['WT']
            for candidate in candidates:
                candidate.ph_opt = ph[candidate.mutation]
                candidate.ph_uncertainty = ph_details[candidate.mutation]['uncertainty']
                candidate.delta_ph_from_wt = candidate.ph_opt - wt_ph
            report["stages"].append({
                "stage": STAGES[3],
                "wt_ph": wt_ph,
                "count": len(candidates),
            })

            write_candidates_csv(run_dir / 'scorer_predictions.csv', candidates)

            # 4) UniStab: batch input in original candidate order.
            ddg_details = self.unistab.predict_ddg_with_uncertainty(
                seq,
                {
                    c.mutation: c.mutant_sequence
                    for c in candidates
                    if c.mutant_sequence
                },
                work_dir=run_dir / "unistab",
                env=env,
                timeout=timeout,
                force_cpu=self.force_cpu,
            )
            ddg = {name: value['ddg'] for name,value in ddg_details.items()}
            artifact['stability_predictions'] = ddg_details
            write_json(run_dir / 'bioartifact.json', artifact.to_dict())
            for candidate in candidates:
                candidate.ddg = ddg[candidate.mutation]
                candidate.stability_uncertainty = ddg_details[candidate.mutation]['uncertainty']
            report["stages"].append({"stage": STAGES[4], "count": len(candidates)})

            # Persist the same fixed-reference components used by server and browser.
            if cfg['selection'].get('score_policy') is not None:
                from .core.screening import freeze_reference_policy
                reference = cfg['selection'].get('reference') or {'activity_log10': activity['WT']['mu_log10'], 'ph_opt': wt_ph}
                cfg['selection']['score_policy'] = freeze_reference_policy(candidates, cfg['selection'], reference)
                cfg['selection']['reference'] = reference
                write_json(run_dir / 'scoring_reference.json', {
                    'reference': reference, 'score_policy': cfg['selection']['score_policy']})
            selection_file = run_dir / 'selection.yaml'
            accepted_snapshot = selection_file.read_text() if selection_file.exists() else None
            if accepted_snapshot is not None:
                accepted = yaml.safe_load(accepted_snapshot)
                accepted['score_policy'] = cfg['selection']['score_policy']
                if 'reference' in cfg['selection']:
                    accepted['reference'] = cfg['selection']['reference']
                cfg['selection'] = accepted
            if llm_cfg['enabled'] and llm_cfg.get('auto_balance', False):
                from .core.screening import summarize_scorer_evidence
                from .llm.gemini import MiniMaxClient
                from .tools.builtin import BALANCE_TOOL_SCHEMA
                client = self._llm_client(cfg)
                if not isinstance(client, MiniMaxClient):
                    raise ValueError('Automatic native tool balancing requires MiniMax')
                evidence = summarize_scorer_evidence([c.to_dict() for c in candidates])
                previous = copy.deepcopy(cfg['selection'])
                applied = []
                def apply_balance(name, arguments):
                    if name != 'balance_scores' or set(arguments) != {'weights','confidence'}:
                        raise ValueError('Invalid balancing tool call')
                    if arguments['weights'] != previous['weights']:
                        raise ValueError('Automatic balancing must preserve user preferences')
                    weights = balance_weights(arguments['weights'],arguments['confidence'])
                    cfg['selection']['confidence'] = arguments['confidence']
                    applied.append(weights)
                    return {'effective_weights': weights, 'confidence': arguments['confidence']}
                decision = client.tool_loop(model=llm_cfg['parameter_model'], max_tokens=llm_cfg['max_tokens'],
                    tools=[BALANCE_TOOL_SCHEMA], execute=apply_balance,
                    system_prompt="""You control Enzora scorer reliability. Call balance_scores exactly once.
Preserve all user preference weights exactly. Use measured uncertainty evidence and the frozen
uncertainty scales to choose confidence controls in [0,1]; explain each choice and units.
Compare uncertainty only after normalization by its own model scale. Unknown uncertainty is
unknown, never high/low confidence. These are reliability controls, not calibrated probabilities.
The engine also applies each candidate's measured uncertainty through
r_ki = 1 / (1 + u_ki / tau_i). Your controls define a run policy, not identical
final influence for all candidates. Never change activity, pH or stability predictions.
Never claim a higher validated accuracy. After tool output, explain the policy and
candidate-specific uncertainty adjustment; do not describe global weights as candidate confidence.""",
                    user_prompt=json.dumps({'selection':previous,'scorer_evidence':evidence},ensure_ascii=False))
                if len(applied) != 1:
                    raise ValueError('Assistant must apply exactly one confidence balance')
                with (run_dir / 'llm_balance_decisions.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps({'provider':'minimax','model':llm_cfg['parameter_model'],
                        'reason':decision['message'],'previous_selection':previous,
                        'scorer_evidence':evidence,'calls':decision['calls']},ensure_ascii=False,allow_nan=False)+'\n')
            if selection_file.exists() and selection_file.read_text() != accepted_snapshot:
                raise ValueError('Selection changed during automatic balancing; review and retry the run')
            temporary_selection = selection_file.with_suffix('.yaml.tmp')
            temporary_selection.write_text(yaml.safe_dump(cfg['selection']))
            temporary_selection.replace(selection_file)
            rank_candidates(candidates, cfg['selection'])
            write_candidates_csv(run_dir / 'scorer_predictions.csv', candidates)
            self._phase(run_dir, 3, cfg)
            cfg['selection'] = yaml.safe_load(selection_file.read_text())
            validate_candidate_and_selection(cfg, strict_ph_target=True)
            # Final selection uses the last explicitly accepted phase-2 controls.
            effective_weights = balance_weights(cfg['selection']['weights'], cfg['selection']['confidence'])
            write_json(run_dir / 'balancing.json', {'preferences': cfg['selection']['weights'],
                'confidence': cfg['selection']['confidence'], 'effective_weights': effective_weights,
                'meaning': 'Explicit reliability controls, not calibrated model probabilities'})
            ranked = rank_candidates(candidates, cfg['selection'])
            report["stages"].append({
                "stage": STAGES[5],
                "passed": sum(1 for c in ranked if c.passed),
            })

            # 6) Add 3-objective Pareto status after normalized scoring components exist.
            mark_pareto_optimal(ranked)
            passed = [c for c in ranked if c.passed]
            selected = passed[: int(cfg["selection"]["top_k"])]
            report["stages"].append({
                "stage": STAGES[6],
                "feature_count": len(ranked),
                "pareto_optimal": sum(1 for c in ranked if c.pareto_optimal),
                "selected": len(selected),
            })

            ranked_csv = run_dir / "02_candidate_pool_ranked.csv"
            selected_csv = run_dir / "03_selected_topk.csv"
            features_csv = run_dir / "04_mutation_features.csv"
            write_candidates_csv(ranked_csv, ranked)
            write_candidates_csv(selected_csv, selected)
            write_mutation_features_csv(features_csv, ranked)

            # 7) Optional codon optimization is a deterministic post-processing step.
            # Built-in hosts are offline; non-built-in hosts can query/cache Kazusa.
            # It runs only when the user explicitly mentions codon optimization.
            codon_info = write_codon_optimization_outputs(
                run_dir=run_dir,
                sequence_name=str(cfg.get("sequence_name", "enzyme")),
                wt_sequence=seq,
                selected=selected,
                resolved_request=codon_request,
                cache_dir=cfg.get("runtime", {}).get("cache_dir"),
            )
            report["stages"].append({
                "stage": STAGES[7],
                "enabled": bool(codon_info.get("enabled")),
                "status": codon_info.get("status"),
                "host_count": codon_info.get("host_count", 0),
                "organisms": [
                    host.get("scientific_name")
                    for host in codon_info.get("resolved_hosts", [])
                ],
                "count": codon_info.get("count", 0),
                "failed_hosts": codon_info.get("failed_hosts", []),
                "warnings": codon_info.get("warnings", []),
            })

            # 8) Existing plots + mutation sequence map + Pareto plot.
            if cfg['visualization']['umap']:
                from .outputs.visualization import write_sequence_umap
                write_sequence_umap(candidates, selected, run_dir / 'umap.json', cfg['mutation']['random_seed'])
            figure_paths: list[Path] = []
            if bool(cfg.get("visualization", {}).get("enabled", True)):
                figure_paths = generate_visualizations(
                    ranked_csv=ranked_csv,
                    selected_csv=selected_csv,
                    output_dir=run_dir / "figures",
                    sequence_length=len(seq),
                )
            report["stages"].append({
                "stage": STAGES[8],
                "enabled": bool(cfg.get("visualization", {}).get("enabled", True)),
                "figures": [str(p) for p in figure_paths],
            })

            # 8) LLM now also receives deterministic mutation descriptors/Pareto state.
            analysis_text = (
                "LLM 结果分析未启用。"
                if report_language == "zh"
                else "LLM result analysis is disabled."
            )
            analysis_error: str | None = None
            if bool(llm_cfg.get("enabled", False)) and bool(llm_cfg.get("result_analysis", True)):
                from .llm.result_analyzer import analyze_results

                try:
                    client = self._llm_client(cfg)
                    analysis_text = analyze_results(
                        cfg=cfg,
                        ranked_csv=ranked_csv,
                        wt_ph=wt_ph,
                        client=client,
                    )
                    if self.print_llm:
                        print("\n[LLM 结果分析]")
                        print(analysis_text)
                except Exception as exc:
                    analysis_error = f"{type(exc).__name__}: {exc}"
                    if not bool(llm_cfg.get("analysis_fail_open", True)):
                        raise
                    if report_language == "zh":
                        analysis_text = (
                            "LLM 结果分析调用失败，但三个预测模型、排序、点突变分析与可视化结果已经保留。\n"
                            f"错误: {analysis_error}"
                        )
                    else:
                        analysis_text = (
                            "LLM result analysis failed, but the three-model predictions, ranking, "
                            "point-mutation analysis and visualizations were retained.\n"
                            f"Error: {analysis_error}"
                        )
            (run_dir / "analysis.txt").write_text(analysis_text, encoding="utf-8")
            report["stages"].append({
                "stage": STAGES[9],
                "enabled": bool(llm_cfg.get("enabled", False) and llm_cfg.get("result_analysis", True)),
                "error": analysis_error,
            })

            # 9) Emit exactly one chosen Markdown language per run.
            report_md = build_markdown_report(
                run_dir=run_dir,
                analysis_text=analysis_text,
                figure_paths=figure_paths,
                selected_csv=selected_csv,
                user_request=cfg.get("user_request"),
                language=report_language,
                codon_optimization=codon_info,
            )
            report["stages"].append({"stage": STAGES[10], "file": str(report_md)})

            report["effective_weights"] = effective_weights
            report["selected"] = [c.to_dict() for c in selected]
            report["candidate_pool_size"] = len(candidates)
            report["wt_ph"] = wt_ph
            report["mutation_features_file"] = str(features_csv)
            report["codon_optimization"] = codon_info
            report["analysis_file"] = str(run_dir / "analysis.txt")
            report["markdown_report"] = str(report_md)
            report["figures"] = [str(p) for p in figure_paths]
            write_json(run_dir / "progress.json", {"phase": 3, "status": "success"})
            report["status"] = "success"
            write_json(run_dir / "result.json", report)
            return report
        except Exception as exc:
            report["status"] = "failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
            write_json(run_dir / "result.json", report)
            (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        finally:
            # Prediction tables, model logs and generation telemetry remain available.
            if not cfg['runtime']['keep_temp']:
                try:
                    cleanup = cleanup_run_cache(run_dir)
                    if cleanup['errors']:
                        print('Cache cleanup requires attention: ' + '; '.join(cleanup['errors']))
                except OSError as exc:
                    print(f'Cache cleanup could not finish: {type(exc).__name__}: {exc}')
            print(f"本次运行目录: {run_dir}")
