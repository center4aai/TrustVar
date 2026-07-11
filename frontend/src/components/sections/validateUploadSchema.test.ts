import { describe, it, expect } from 'vitest';
import { validateUploadSchema } from './DatasetUploadModal';

const base = { promptColumn: 'prompt', targetColumn: 'target' };

describe('validateUploadSchema (N1)', () => {
  it('rejects missing prompt column', () => {
    const errs = validateUploadSchema({
      taskType: 'generation', columns: ['x'], promptColumn: 'prompt', targetColumn: '',
    });
    expect(errs.some((e) => e.includes('Prompt column'))).toBe(true);
  });

  it('open_qa requires target', () => {
    const errs = validateUploadSchema({
      taskType: 'open_qa', columns: ['prompt'], promptColumn: 'prompt', targetColumn: '',
    });
    expect(errs.some((e) => e.includes('requires a Target Column'))).toBe(true);
  });

  it('classification requires target', () => {
    const errs = validateUploadSchema({
      taskType: 'classification', columns: ['prompt'], promptColumn: 'prompt', targetColumn: '',
    });
    expect(errs.some((e) => e.includes('requires a Target Column'))).toBe(true);
  });

  it('mcq warns when no options column present', () => {
    const errs = validateUploadSchema({
      taskType: 'mcq', columns: ['prompt', 'target'], ...base,
    });
    expect(errs.some((e) => e.includes('answer options'))).toBe(true);
  });

  it('passes when columns present', () => {
    expect(validateUploadSchema({
      taskType: 'open_qa', columns: ['prompt', 'target'], ...base,
    })).toEqual([]);
  });
});

describe('validateUploadSchema (D1)', () => {
  it('generation does NOT require a Target column (D1)', () => {
    const errs = validateUploadSchema({
      taskType: 'generation', columns: ['prompt'], promptColumn: 'prompt', targetColumn: '',
    });
    expect(errs.some((e) => e.includes('requires a Target Column'))).toBe(false);
  });
});
