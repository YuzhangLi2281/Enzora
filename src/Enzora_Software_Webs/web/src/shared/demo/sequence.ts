/** Explicit sequence parsing and deterministic translation for demo data. */

import type { TranslationResult } from '../api/contracts';

const STOP = new Set(['TAA', 'TAG', 'TGA']);

const CODON_TABLE: Record<string, string> = {
  TTT: 'F', TTC: 'F', TTA: 'L', TTG: 'L', TCT: 'S', TCC: 'S', TCA: 'S', TCG: 'S',
  TAT: 'Y', TAC: 'Y', TAA: '*', TAG: '*', TGT: 'C', TGC: 'C', TGA: '*', TGG: 'W',
  CTT: 'L', CTC: 'L', CTA: 'L', CTG: 'L', CCT: 'P', CCC: 'P', CCA: 'P', CCG: 'P',
  CAT: 'H', CAC: 'H', CAA: 'Q', CAG: 'Q', CGT: 'R', CGC: 'R', CGA: 'R', CGG: 'R',
  ATT: 'I', ATC: 'I', ATA: 'I', ATG: 'M', ACT: 'T', ACC: 'T', ACA: 'T', ACG: 'T',
  AAT: 'N', AAC: 'N', AAA: 'K', AAG: 'K', AGT: 'S', AGC: 'S', AGA: 'R', AGG: 'R',
  GTT: 'V', GTC: 'V', GTA: 'V', GTG: 'V', GCT: 'A', GCC: 'A', GCA: 'A', GCG: 'A',
  GAT: 'D', GAC: 'D', GAA: 'E', GAG: 'E', GGT: 'G', GGC: 'G', GGA: 'G', GGG: 'G',
};

export function cleanSequence(text: string): { body: string; name: string | null } {
  let raw = (text || '').trim();
  let name: string | null = null;
  if (raw.startsWith('>')) {
    const lines = raw.split(/\r?\n/);
    name = (lines[0].slice(1).trim() || null);
    raw = lines.slice(1).join('');
  }
  return { body: raw.replace(/\s+/g, '').toUpperCase(), name };
}

export function translateNucleotide(sequence: string): TranslationResult {
  const dna = sequence.replace(/U/g, 'T').toUpperCase().replace(/\s+/g, '');
  const warnings: string[] = [];
  let start = dna.indexOf('ATG');
  if (start < 0) {
    start = 0;
    warnings.push('no ATG start codon; translated from position 1');
  }
  let body = dna.slice(start);
  if (body.length % 3) {
    warnings.push('length not a multiple of 3; trailing bases trimmed');
    body = body.slice(0, body.length - (body.length % 3));
  }
  let amino = '';
  let stopIndex: number | null = null;
  for (let i = 0; i < body.length; i += 3) {
    const codon = body.slice(i, i + 3);
    if (STOP.has(codon)) { stopIndex = i / 3; break; }
    amino += CODON_TABLE[codon] ?? 'X';
  }
  if (stopIndex === null) warnings.push('no in-frame stop codon; translated to end');
  return {
    protein_sequence: amino,
    orf_start_1based: start + 1,
    orf_end_1based: start + (stopIndex !== null ? stopIndex * 3 : body.length),
    cds_length: body.length,
    warnings,
  };
}

export function applyMutation(sequence: string, mutation: string): string {
  const m = mutation.match(/^([A-Z])(\d+)([A-Z])$/);
  if (!m) throw new Error(`bad mutation format: ${mutation}`);
  const pos = Number(m[2]) - 1;
  if (pos < 0 || pos >= sequence.length) throw new Error(`position out of range: ${mutation}`);
  return sequence.slice(0, pos) + m[3] + sequence.slice(pos + 1);
}
