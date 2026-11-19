import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { AlertCircle, Download, FileUp, ListChecks, Play, RefreshCw, Sparkles, X } from 'lucide-react';
import { useProject } from '../contexts/ProjectContext';
import { useAzureCredentials, hasAzureCredentials } from '../contexts/AzureCredentialsContext';
import {
  getMockAutStatus,
  runProjectVectorChecks,
  runProjectWorkbookChecks,
  suggestOffTopicQuestions,
} from '../api/client';
import { downloadBase64File } from '../api/workbench';
import type { MockAutStatus } from '../api/types';

const CHECKS = [
  { id: 'vector_integrity', label: 'Vector integrity', description: 'Without a workbook, check collection-wide dimension consistency; with a workbook, check dimensions and basic vector validity in the top three retrieved records per question.' },
  { id: 'neighbor_consistency', label: 'Neighbor consistency', description: 'For each workbook question, retrieve top-k chunks, check every distinct pair, and report the pair results.' },
  { id: 'aut_vector_drift', label: 'AUT-Vector Drift', description: 'For each workbook question, retrieve top-k vector records, get the AUT answer from those records, then judge it against the question and evidence.' },
  { id: 'duplicate_detection', label: 'Duplicate similarity candidates', description: 'For each workbook question, compare the retrieved records and flag high-similarity pairs.' },
  { id: 'metadata_completeness', label: 'Metadata completeness', description: 'Without a workbook, check required fields across the collection; with a workbook, inspect the top three retrieved records per question.' },
  { id: 'negative_query_similarity', label: 'Negative queries (off-topic)', description: 'Check that unrelated questions receive low nearest-chunk cosine scores.' },
  { id: 'hit_at_k', label: 'Hit@k (per question)', description: 'For each workbook question, retrieve the top 50 text, table, image, and graph records and report the correct item’s rank plus Hit@1/3/5/10.' },
] as const;

const OFF_TOPIC_CATEGORIES = [
  {
    id: 'insurance',
    label: 'Insurance',
    questions: [
      'What is the difference between term and whole life insurance?',
      'How does a deductible affect an insurance claim?',
      'What information is needed to get a home insurance quote?',
      'Does travel insurance cover a canceled flight?',
      'How can I add a new driver to my auto insurance policy?',
    ],
  },
  {
    id: 'finance',
    label: 'Personal finance & banking',
    questions: [
      'How does compound interest work in a savings account?',
      'What should I compare when choosing a credit card?',
      'How do I create a monthly household budget?',
      'What is the difference between a debit card and a credit card?',
      'How can I improve my credit score?',
    ],
  },
  {
    id: 'healthcare',
    label: 'Healthcare & wellness',
    questions: [
      'How many hours of sleep do adults generally need?',
      'What are some beginner-friendly ways to stay active?',
      'How should I prepare for a routine medical appointment?',
      'What foods are good sources of dietary fiber?',
      'How can I build a consistent daily meditation habit?',
    ],
  },
  {
    id: 'technology',
    label: 'Technology & computing',
    questions: [
      'How can I back up photos from my phone to a computer?',
      'What is the difference between Wi-Fi and mobile data?',
      'Which laptop specifications matter for everyday use?',
      'How do I create a strong password for an online account?',
      'What are common uses for a smart home device?',
    ],
  },
  {
    id: 'education',
    label: 'Education & careers',
    questions: [
      'What are effective ways to prepare for a job interview?',
      'How can I organize a weekly study schedule?',
      'What should I include in a professional resume?',
      'How do online courses compare with in-person classes?',
      'What skills are useful when changing careers?',
    ],
  },
  {
    id: 'law-government',
    label: 'Law & government',
    questions: [
      'How do I find the official website for my local government?',
      'What documents are commonly needed to renew a passport?',
      'How does jury duty work?',
      'Where can I find information about local voter registration?',
      'What is the general process for applying for a building permit?',
    ],
  },
  {
    id: 'real-estate',
    label: 'Real estate & housing',
    questions: [
      'What costs should I consider when buying a first home?',
      'How is monthly rent typically determined?',
      'What should I look for during a rental property inspection?',
      'How do fixed-rate and adjustable-rate mortgages differ?',
      'What questions should I ask before signing a lease?',
    ],
  },
  {
    id: 'automotive',
    label: 'Automotive & transportation',
    questions: [
      'How often should a car engine oil be changed?',
      'What should I keep in an emergency roadside kit?',
      'How can I compare public transit passes in a city?',
      'What are signs that a car battery needs replacing?',
      'How do I plan a fuel-efficient road trip?',
    ],
  },
  {
    id: 'travel',
    label: 'Travel & tourism',
    questions: [
      'What are some popular places to visit in Kyoto?',
      'What should I pack for a week-long beach vacation?',
      'When is the best season to see autumn leaves in New England?',
      'How can I find budget-friendly flights to Rome?',
      'What documents are commonly needed for international travel?',
    ],
  },
  {
    id: 'sports',
    label: 'Sports & recreation',
    questions: [
      'How many players from one team are on a soccer field?',
      'What is the standard distance of a marathon?',
      'How does the scoring system work in tennis?',
      'Which equipment is used to play ice hockey?',
      'How often are the Summer Olympic Games held?',
    ],
  },
  {
    id: 'food',
    label: 'Food & cooking',
    questions: [
      'What temperature should I use to roast vegetables?',
      'How do I make a basic tomato soup?',
      'What can I substitute for eggs in a cake?',
      'How long should I boil dried pasta?',
      'Which herbs pair well with roasted potatoes?',
    ],
  },
  {
    id: 'entertainment',
    label: 'Entertainment & media',
    questions: [
      'Who directed the movie Jurassic Park?',
      'What are some popular board games for families?',
      'How do I tune a guitar?',
      'Which musical instruments are common in a jazz band?',
      'Can you recommend a light comedy movie for a weekend?',
    ],
  },
  {
    id: 'home-garden',
    label: 'Home, pets & gardening',
    questions: [
      'How often should I water a typical houseplant?',
      'What toys are suitable for an indoor cat?',
      'Which vegetables grow well in a small container garden?',
      'How can I teach a dog to respond to its name?',
      'What kind of soil is best for growing tomatoes?',
    ],
  },
] as const;

type WorkbookReport = Awaited<ReturnType<typeof runProjectWorkbookChecks>>;
type Report = Awaited<ReturnType<typeof runProjectVectorChecks>>
  & Partial<Pick<WorkbookReport, 'workbook_questions'>>;

interface SavedExecutionSettings {
  selectedChecks: string[];
  requiredFields: string;
  duplicateMin: number;
  duplicateMax: number;
  driftMaxScore: number;
  questionRetrievalTopK: number;
  selectedNegativeQuestions: string[];
  selectedSuggestedQuestions: string[];
  suggestionTopics: string;
  suggestionCount: number;
  suggestions: Array<{ topic: string; questions: string[] }>;
  negativeThreshold: number;
}

function readExecutionSettings(projectId: string): SavedExecutionSettings | null {
  const saved = localStorage.getItem(`rag_tester_execution_settings_${projectId}`);
  if (!saved) return null;
  const value: unknown = JSON.parse(saved);
  if (!value || typeof value !== 'object') {
    throw new Error('Saved execution settings are invalid.');
  }
  const settings = value as Partial<SavedExecutionSettings>;
  if (
    !Array.isArray(settings.selectedChecks)
    || !settings.selectedChecks.every((item) => typeof item === 'string')
    || !Array.isArray(settings.selectedNegativeQuestions)
    || !settings.selectedNegativeQuestions.every((item) => typeof item === 'string')
    || !Array.isArray(settings.selectedSuggestedQuestions)
    || !settings.selectedSuggestedQuestions.every((item) => typeof item === 'string')
    || !Array.isArray(settings.suggestions)
    || !settings.suggestions.every((item) => (
      Boolean(item)
      && typeof item === 'object'
      && typeof item.topic === 'string'
      && Array.isArray(item.questions)
      && item.questions.every((question) => typeof question === 'string')
    ))
    || typeof settings.requiredFields !== 'string'
    || typeof settings.suggestionTopics !== 'string'
    || typeof settings.duplicateMin !== 'number'
    || typeof settings.duplicateMax !== 'number'
    || (
      settings.driftMaxScore !== undefined
      && (
        typeof settings.driftMaxScore !== 'number'
        || !Number.isFinite(settings.driftMaxScore)
        || settings.driftMaxScore < 0
        || settings.driftMaxScore > 1
      )
    )
    || (
      settings.questionRetrievalTopK !== undefined
      && (
        typeof settings.questionRetrievalTopK !== 'number'
        || !Number.isInteger(settings.questionRetrievalTopK)
        || settings.questionRetrievalTopK < 1
        || settings.questionRetrievalTopK > 20
      )
    )
    || typeof settings.suggestionCount !== 'number'
    || typeof settings.negativeThreshold !== 'number'
  ) {
    throw new Error('Saved execution settings are invalid.');
  }
  const availableChecks = new Set<string>(CHECKS.map(({ id }) => id));
  return {
    ...settings,
    driftMaxScore: settings.driftMaxScore ?? 0.25,
    questionRetrievalTopK: settings.questionRetrievalTopK ?? 5,
    selectedChecks: settings.selectedChecks.filter((check) => availableChecks.has(check)),
  } as SavedExecutionSettings;
}

function downloadHtml(html: string, filename: string) {
  const url = URL.createObjectURL(new Blob([html], { type: 'text/html' }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `${filename}.html`;
  anchor.click();
  URL.revokeObjectURL(url);
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function asRecords(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value)
    ? value.map(asRecord).filter((item): item is Record<string, unknown> => item !== null)
    : [];
}

function CheckStatus({ value, label }: { value: string; label?: string }) {
  const statusClass = value === 'PASS' || value === 'MUTUAL' || value === 'CLEAR'
    ? 'question-check-status-pass'
    : value === 'DRIFT' || value === 'MISMATCH' || value === 'REVIEW'
      ? 'question-check-status-warn'
      : value === 'ERROR' || value === 'FAIL'
        ? 'question-check-status-error'
        : 'question-check-status-info';
  return <span className={`question-check-status ${statusClass}`}>{label ?? value}</span>;
}

function HitAtKResults({ result }: { result: Record<string, unknown> }) {
  const detail = asRecord(result.detail);
  const rows = asRecords(detail?.question_results);
  if (!detail) return null;

  return (
    <section className="question-check-results">
      <div className="question-check-heading">
        <div>
          <h3>Hit@k by question</h3>
          <p>Top {String(detail.top_k ?? 50)} retrieved records; no averages.</p>
        </div>
      </div>
      {rows.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Question</th>
                <th>Type</th>
                <th>Rank</th>
                <th>Hit@1</th>
                <th>Hit@3</th>
                <th>Hit@5</th>
                <th>Hit@10</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={`${String(row.row ?? index)}-${index}`}>
                  <td>{String(row.question ?? '')}</td>
                  <td>{String(row.type ?? '')}</td>
                  <td>{String(row.rank ?? 'not found')}</td>
                  <td>{String(row['hit@1'] ?? 0)}</td>
                  <td>{String(row['hit@3'] ?? 0)}</td>
                  <td>{String(row['hit@5'] ?? 0)}</td>
                  <td>{String(row['hit@10'] ?? 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="question-check-empty">No per-question Hit@k results were returned.</p>
      )}
    </section>
  );
}

function CheckResultModal({
  title,
  summary,
  eyebrow = 'Check details',
  description = 'Full result, evidence, and record-level information.',
  children,
}: {
  title: string;
  summary: ReactNode;
  eyebrow?: string;
  description?: string;
  children: ReactNode;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (isOpen && !dialog.open) dialog.showModal();
    if (!isOpen && dialog.open) dialog.close();
  }, [isOpen]);

  return (
    <>
      <button
        className="question-check-trigger"
        type="button"
        aria-haspopup="dialog"
        onClick={() => setIsOpen(true)}
      >
        <span className="question-check-trigger-summary">{summary}</span>
        <span className="question-check-trigger-action">View full details</span>
      </button>
      <dialog
        ref={dialogRef}
        className="check-result-dialog"
        aria-label={`${title} details`}
        onClose={() => setIsOpen(false)}
        onClick={(event) => {
          if (event.target === event.currentTarget) setIsOpen(false);
        }}
      >
        <section className="check-result-modal">
          <header className="check-result-modal-header">
            <div>
              <p className="check-result-modal-eyebrow">{eyebrow}</p>
              <h2>{title}</h2>
              <p>{description}</p>
            </div>
            <button
              className="check-result-modal-close"
              type="button"
              aria-label="Close check details"
              onClick={() => setIsOpen(false)}
            >
              <X size={17} />
            </button>
          </header>
          <div className="check-result-modal-body">{children}</div>
        </section>
      </dialog>
    </>
  );
}

interface QuestionCheckRow {
  questionIndex: number;
  question: string;
  neighbor?: Record<string, unknown>;
  duplicate?: Record<string, unknown>;
  drift?: Record<string, unknown>;
  vectorIntegrity?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
  workbook?: Record<string, unknown>;
}

function QuestionWiseResults({
  report,
  selectedChecks,
}: {
  report: Report;
  selectedChecks: string[];
}) {
  const neighborDetail = asRecord(
    report.results.find((result) => result.check === 'known_similar_pairs')?.detail,
  );
  const duplicateDetail = asRecord(
    report.results.find((result) => result.check === 'near_duplicate_detection_by_question')?.detail,
  );
  const driftDetail = asRecord(
    report.results.find((result) => result.check === 'AUT-Vector Drift')?.detail,
  );
  const vectorIntegrityDetail = asRecord(
    report.results.find((result) => result.check === 'dimension_consistency_by_question')?.detail,
  );
  const metadataDetail = asRecord(
    report.results.find((result) => result.check === 'metadata_completeness_by_question')?.detail,
  );
  const rowsByIndex = new Map<number, QuestionCheckRow>();
  const getRow = (questionIndex: number, question: string) => {
    const row = rowsByIndex.get(questionIndex) ?? {
      questionIndex,
      question,
    };
    if (!row.question && question) row.question = question;
    rowsByIndex.set(questionIndex, row);
    return row;
  };

  asRecords(neighborDetail?.retrievals_by_question).forEach((item, index) => {
    const questionIndex = typeof item.question_index === 'number' ? item.question_index : index;
    const row = getRow(questionIndex, String(item.question ?? ''));
    row.neighbor = item;
  });
  asRecords(duplicateDetail?.question_results).forEach((item, index) => {
    const questionIndex = typeof item.question_index === 'number' ? item.question_index : index;
    const row = getRow(questionIndex, String(item.question ?? ''));
    row.duplicate = item;
  });
  asRecords(driftDetail?.question_results).forEach((item, index) => {
    const questionIndex = typeof item.question_index === 'number' ? item.question_index : index;
    const row = getRow(questionIndex, String(item.question ?? ''));
    row.drift = item;
  });
  asRecords(vectorIntegrityDetail?.question_results).forEach((item, index) => {
    const questionIndex = typeof item.question_index === 'number' ? item.question_index : index;
    const row = getRow(questionIndex, String(item.question ?? ''));
    row.vectorIntegrity = item;
  });
  asRecords(metadataDetail?.question_results).forEach((item, index) => {
    const questionIndex = typeof item.question_index === 'number' ? item.question_index : index;
    const row = getRow(questionIndex, String(item.question ?? ''));
    row.metadata = item;
  });
  asRecords(report.workbook_questions).forEach((item, index) => {
    const questionIndex = typeof item.question_index === 'number' ? item.question_index : index;
    const row = getRow(questionIndex, String(item.question ?? ''));
    row.workbook = item;
  });

  const rows = [...rowsByIndex.values()].sort((a, b) => a.questionIndex - b.questionIndex);
  const neighborMisses = rows.reduce((total, row) => {
    const check = asRecord(row.neighbor?.check_result);
    const detail = asRecord(check?.detail);
    return total + asRecords(detail?.pair_results)
      .filter((pair) => pair.status === 'one_way' || pair.status === 'neither').length;
  }, 0);
  const duplicateCandidates = rows.reduce((total, row) => (
    total + (typeof row.duplicate?.candidate_pairs_found === 'number'
      ? row.duplicate.candidate_pairs_found
      : 0)
  ), 0);
  const driftFailures = rows.filter((row) => row.drift?.status === 'DRIFT').length;
  const driftErrors = rows.filter((row) => row.drift?.status === 'ERROR').length;
  const vectorIntegrityFailures = rows.filter((row) => row.vectorIntegrity?.status === 'FAIL').length;
  const metadataIssues = rows.reduce((total, row) => (
    total + (typeof row.metadata?.total_issues === 'number' ? row.metadata.total_issues : 0)
  ), 0);
  const duplicateMinimum = typeof duplicateDetail?.minimum_similarity === 'number'
    ? duplicateDetail.minimum_similarity.toFixed(2)
    : '—';
  const duplicateMaximum = typeof duplicateDetail?.maximum_similarity === 'number'
    ? duplicateDetail.maximum_similarity.toFixed(2)
    : '—';
  const visibleChecks = [
    selectedChecks.includes('neighbor_consistency') ? 'neighbor' : null,
    selectedChecks.includes('duplicate_detection') ? 'duplicate' : null,
    selectedChecks.includes('aut_vector_drift') ? 'drift' : null,
    selectedChecks.includes('vector_integrity') && vectorIntegrityDetail ? 'vectorIntegrity' : null,
    selectedChecks.includes('metadata_completeness') && metadataDetail ? 'metadata' : null,
  ].filter((check): check is 'neighbor' | 'duplicate' | 'drift' | 'vectorIntegrity' | 'metadata' => check !== null);

  if (!visibleChecks.length) return null;

  return (
    <section className="question-check-results">
      <div className="question-check-heading">
        <div>
          <p className="question-check-eyebrow">Workbook evaluation</p>
          <h3>Results by question</h3>
          <p>Select a question to view its answers and source details, or open a check for its evidence and record-level results.</p>
        </div>
        <span className="question-check-count">{rows.length} question{rows.length === 1 ? '' : 's'}</span>
      </div>
      <div className="question-check-summary">
        {visibleChecks.includes('neighbor') && (
          <div className="question-check-stat">
            <span>Neighbor ranking misses</span>
            <strong>{neighborMisses}</strong>
          </div>
        )}
        {visibleChecks.includes('duplicate') && (
          <div className="question-check-stat">
            <span>Duplicate candidates · cosine range</span>
            <strong>{duplicateCandidates} <small>{duplicateMinimum}–{duplicateMaximum}</small></strong>
          </div>
        )}
        {visibleChecks.includes('drift') && (
          <>
            <div className="question-check-stat">
              <span>Questions with drift</span>
              <strong>{driftFailures}</strong>
            </div>
            <div className="question-check-stat">
              <span>Drift errors</span>
              <strong>{driftErrors}</strong>
            </div>
          </>
        )}
        {visibleChecks.includes('vectorIntegrity') && (
          <div className="question-check-stat">
            <span>Questions with vector issues</span>
            <strong>{vectorIntegrityFailures}</strong>
          </div>
        )}
        {visibleChecks.includes('metadata') && (
          <div className="question-check-stat">
            <span>Incomplete required fields</span>
            <strong>{metadataIssues}</strong>
          </div>
        )}
      </div>
      {rows.length ? (
        <div className="question-check-table-wrap">
          <table className="question-check-table">
            <thead>
              <tr>
                <th className="question-check-question-head">Workbook question</th>
                {visibleChecks.includes('neighbor') && <th>Neighbor consistency</th>}
                {visibleChecks.includes('duplicate') && <th>Duplicate candidates</th>}
                {visibleChecks.includes('drift') && <th>AUT-Vector Drift</th>}
                {visibleChecks.includes('vectorIntegrity') && <th>Vector integrity</th>}
                {visibleChecks.includes('metadata') && <th>Metadata completeness</th>}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.questionIndex}>
                  <td className="question-check-question">
                    <QuestionDetailsModal row={row} />
                  </td>
                  {visibleChecks.includes('neighbor') && <td><NeighborQuestionResult result={row.neighbor} /></td>}
                  {visibleChecks.includes('duplicate') && <td><DuplicateQuestionResult result={row.duplicate} /></td>}
                  {visibleChecks.includes('drift') && <td><DriftQuestionResult result={row.drift} /></td>}
                  {visibleChecks.includes('vectorIntegrity') && <td><VectorIntegrityQuestionResult result={row.vectorIntegrity} /></td>}
                  {visibleChecks.includes('metadata') && <td><MetadataQuestionResult result={row.metadata} /></td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="question-check-empty">No per-question results were returned for this workbook.</p>
      )}
      <p className="question-check-note">
        Vector integrity and metadata completeness inspect the top three retrieved records for each question. Neighbor and duplicate checks use their configured top-k; duplicate results are similarity candidates only.
      </p>
    </section>
  );
}

function displayText(value: unknown): string {
  if (value === null || value === undefined) return '';
  if (Array.isArray(value)) return value.map(String).join(', ');
  return String(value).trim();
}

function displayPages(value: unknown): string {
  if (Array.isArray(value)) return value.map(String).join(', ');
  if (typeof value !== 'string') return displayText(value);
  const text = value.trim();
  if (!text.startsWith('[')) return text;
  try {
    const parsed: unknown = JSON.parse(text);
    return Array.isArray(parsed) ? parsed.map(String).join(', ') : text;
  } catch {
    return text;
  }
}

function QuestionDetailsModal({ row }: { row: QuestionCheckRow }) {
  const workbook = row.workbook ?? {};
  const expectedAnswer = displayText(workbook.expected_answer);
  const driftAnswer = displayText(row.drift?.actual_answer);
  const workbookActualAnswer = displayText(workbook.actual_answer);
  const actualAnswer = driftAnswer || workbookActualAnswer;
  const actualAnswerSource = driftAnswer
    ? 'Generated by AUT in this run'
    : workbookActualAnswer
      ? 'Read from workbook'
      : 'Not generated in this run';
  const retrievedSources = [
    ...asRecords(row.neighbor?.retrieved_sources),
    ...asRecords(row.duplicate?.retrieved_sources),
    ...asRecords(row.vectorIntegrity?.retrieved_sources),
    ...asRecords(row.metadata?.retrieved_sources),
    ...asRecords(row.drift?.retrieved_records).map((record) => {
      const metadata = asRecord(record.metadata) ?? {};
      return {
        id: record.id,
        pdf_name: metadata.pdf_name,
        page_number: metadata.page_number ?? metadata.page_no,
        source_pages: metadata.source_pages ?? metadata.pages,
      };
    }),
  ];
  const sourceKeys = new Set<string>();
  const sources = retrievedSources.filter((source) => {
    const key = [
      displayText(source.pdf_name),
      displayPages(source.page_number ?? source.source_pages),
      displayText(source.id),
    ].join('|');
    if (sourceKeys.has(key)) return false;
    sourceKeys.add(key);
    return true;
  });
  const workbookPdfName = displayText(workbook.pdf_name);
  const workbookPageNumber = displayPages(
    workbook.page_number ?? workbook.page_no ?? workbook.source_pages,
  );
  const pdfNames = [...new Set([
    workbookPdfName,
    ...sources.map((source) => displayText(source.pdf_name)),
  ].filter(Boolean))];
  const pageNumbers = [...new Set([
    workbookPageNumber,
    ...sources.map((source) => displayPages(source.page_number ?? source.source_pages)),
  ].filter(Boolean))];

  return (
    <CheckResultModal
      title={`Question ${row.questionIndex + 1}`}
      eyebrow="Workbook question"
      description="Question, answer comparison, and source information."
      summary={
        <>
          <span className="question-check-question-number">Question {row.questionIndex + 1}</span>
          <span className="question-check-question-summary-text">{row.question || 'Question text unavailable.'}</span>
        </>
      }
    >
      <div className="question-info-layout">
        <section className="question-info-card question-info-question-card">
          <p className="question-info-label">Question</p>
          <p className="question-info-question-text">{row.question || 'Question text unavailable.'}</p>
        </section>
        <div className="question-info-answer-grid">
          <section className="question-info-card">
            <p className="question-info-label">Expected answer</p>
            <p className={expectedAnswer ? 'question-info-answer-text' : 'question-info-placeholder'}>
              {expectedAnswer || 'No expected answer was found in the workbook.'}
            </p>
          </section>
          <section className="question-info-card">
            <div className="question-info-answer-heading">
              <p className="question-info-label">Actual answer</p>
              <span className={actualAnswer ? 'question-info-source-tag' : 'question-info-source-tag question-info-source-tag-muted'}>
                {actualAnswerSource}
              </span>
            </div>
            <p className={actualAnswer ? 'question-info-answer-text' : 'question-info-placeholder'}>
              {actualAnswer || 'Select AUT-Vector Drift to generate an actual answer for this question.'}
            </p>
          </section>
        </div>
        <section className="question-info-card">
          <p className="question-info-label">Source document</p>
          <div className="question-info-source-grid">
            <div>
              <span>PDF name</span>
              <strong>{pdfNames.length ? pdfNames.join(', ') : 'Not available'}</strong>
            </div>
            <div>
              <span>Page number</span>
              <strong>{pageNumbers.length ? pageNumbers.join(', ') : 'Not available'}</strong>
            </div>
          </div>
          {sources.length > 0 && (
            <div className="question-info-source-list">
              {sources.map((source, index) => {
                const pdfName = displayText(source.pdf_name) || 'Retrieved source';
                const pageNumber = displayPages(source.page_number ?? source.source_pages);
                return (
                  <div key={`${displayText(source.id)}-${index}`}>
                    <strong>{pdfName}</strong>
                    <span>
                      {pageNumber ? `Page ${pageNumber}` : 'Page unavailable'}
                      {source.id ? ` · Record ${String(source.id)}` : ''}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      </div>
    </CheckResultModal>
  );
}

function VectorIntegrityQuestionResult({ result }: { result?: Record<string, unknown> }) {
  if (!result) return <span className="question-check-muted">Not run</span>;
  const records = asRecords(result.record_results);
  const issues = asRecords(result.issues);
  const dimensions = asRecord(result.dimensions_found) ?? {};
  const dimensionSummary = Object.entries(dimensions)
    .map(([dimension, count]) => `${dimension}D × ${String(count)}`)
    .join(', ');
  const unavailable = Array.isArray(result.unavailable_records)
    ? result.unavailable_records.map(String)
    : [];
  const status = String(result.status ?? 'INFO');

  return (
    <CheckResultModal
      title="Vector integrity"
      summary={
        <>
          <CheckStatus value={status} />
          <span>{records.length}/{String(result.records_requested ?? 0)} records</span>
        </>
      }
    >
      <div className="question-check-detail">
        <p>Dimensions: {dimensionSummary || 'Unavailable'}</p>
        {records.map((record, index) => (
          <div className="question-check-pair" key={`${String(record.id ?? index)}-${index}`}>
            <strong>{String(record.id ?? `Record ${index + 1}`)} · {String(record.dimension ?? '—')} dimensions</strong>
            {Array.isArray(record.issues) && record.issues.map((issue, issueIndex) => (
              <span key={issueIndex}>{String(issue)}</span>
            ))}
          </div>
        ))}
        {issues.filter((issue) => typeof issue.reason === 'string').map((issue, index) => (
          <p className="question-check-warning" key={`issue-${index}`}>{String(issue.reason)}</p>
        ))}
        {unavailable.length > 0 && <p>Unavailable records: {unavailable.join(', ')}</p>}
      </div>
    </CheckResultModal>
  );
}

function MetadataQuestionResult({ result }: { result?: Record<string, unknown> }) {
  if (!result) return <span className="question-check-muted">Not run</span>;
  const fields = asRecord(result.per_field) ?? {};
  const records = asRecords(result.record_results);
  const status = String(result.status ?? 'INFO');
  const problemRecords = records.filter((record) => asRecords(record.issues).length > 0);

  return (
    <CheckResultModal
      title="Metadata completeness"
      summary={
        <>
          <CheckStatus value={status} />
          <span>{String(result.total_issues ?? 0)} incomplete</span>
        </>
      }
    >
      <div className="question-check-detail">
        <p>Checked {String(result.records_checked ?? 0)} of {String(result.records_requested ?? 0)} retrieved records.</p>
        {Object.entries(fields).map(([field, value]) => {
          const counts = asRecord(value) ?? {};
          return (
            <p key={field}>
              <strong>{field}:</strong> {String(counts.missing ?? 0)} missing, {String(counts.empty ?? 0)} empty
            </p>
          );
        })}
        {problemRecords.map((record, index) => (
          <div className="question-check-pair" key={`${String(record.id ?? index)}-${index}`}>
            <strong>{String(record.id ?? `Record ${index + 1}`)}</strong>
            {asRecords(record.issues).map((issue, issueIndex) => (
              <span key={issueIndex}>{String(issue.field)}: {String(issue.issue)}</span>
            ))}
          </div>
        ))}
        {Array.isArray(result.unavailable_records) && result.unavailable_records.length > 0 && (
          <p>Unavailable records: {result.unavailable_records.map(String).join(', ')}</p>
        )}
      </div>
    </CheckResultModal>
  );
}

function NeighborQuestionResult({ result }: { result?: Record<string, unknown> }) {
  const check = asRecord(result?.check_result);
  const detail = asRecord(check?.detail);
  const pairs = asRecords(detail?.pair_results);
  const mutualPairs = pairs.filter((pair) => pair.status === 'mutual').length;
  const missedPairs = pairs.filter((pair) => pair.status === 'one_way' || pair.status === 'neither').length;
  const unavailablePairs = pairs.filter((pair) => pair.status === 'unavailable').length;
  const status = pairs.length > 0 && mutualPairs === pairs.length
    ? 'MUTUAL'
    : missedPairs > 0
      ? 'MISMATCH'
      : 'INFO';
  const retrievedIds = Array.isArray(result?.retrieved_ids)
    ? result.retrieved_ids.map(String)
    : [];

  if (!result) return <span className="question-check-muted">Not run</span>;
  return (
    <CheckResultModal
      title="Neighbor consistency"
      summary={
        <CheckStatus
          value={status}
          label={`${mutualPairs}/${pairs.length} consistency`}
        />
      }
    >
      <div className="question-check-detail">
        <p>Retrieved records: {retrievedIds.length ? retrievedIds.join(', ') : 'None'}</p>
        <p>{missedPairs} pair{missedPairs === 1 ? '' : 's'} were outside mutual top-k; {unavailablePairs} unavailable.</p>
        {pairs.map((pair, index) => {
          const ids = Array.isArray(pair.pair) ? pair.pair.map(String) : [];
          return (
            <div className="question-check-pair" key={`${ids.join('-')}-${index}`}>
              <div>
                <strong>{ids.join(' ↔ ') || 'Pair'}</strong>
                <CheckStatus value={String(pair.status ?? 'INFO').toUpperCase()} />
              </div>
              <span>Ranks: {String(pair.rank_b_from_a ?? '—')} / {String(pair.rank_a_from_b ?? '—')}</span>
              <span>Boundary overlap: {String(pair.overlap_chars ?? 0)} characters</span>
            </div>
          );
        })}
        {pairs.length === 0 && <p>No record pairs were available to compare.</p>}
      </div>
    </CheckResultModal>
  );
}

function DuplicateQuestionResult({ result }: { result?: Record<string, unknown> }) {
  const candidates = asRecords(result?.candidate_pairs);
  const unavailable = asRecords(result?.unavailable_pairs);
  const status = candidates.length
    ? 'REVIEW'
    : result?.status === 'CLEAR'
      ? 'CLEAR'
      : 'INFO';

  if (!result) return <span className="question-check-muted">Not run</span>;
  return (
    <CheckResultModal
      title="Duplicate detection"
      summary={
        <>
          <CheckStatus value={status} />
          <span>{candidates.length} candidate{candidates.length === 1 ? '' : 's'}</span>
        </>
      }
    >
      <div className="question-check-detail">
        <p>{String(result.pairs_compared ?? 0)} retrieved-record pairs compared.</p>
        {candidates.map((candidate, index) => (
          <details className="question-check-pair" key={`${String(candidate.id_a)}-${String(candidate.id_b)}-${index}`}>
            <summary>
              <strong>{String(candidate.id_a)} ↔ {String(candidate.id_b)}</strong>
              <span>cosine {typeof candidate.cosine_similarity === 'number' ? candidate.cosine_similarity.toFixed(3) : '—'}</span>
            </summary>
            <p>Text token overlap: {typeof candidate.text_token_overlap === 'number' ? `${(candidate.text_token_overlap * 100).toFixed(1)}%` : 'Unavailable'}</p>
            {typeof candidate.chunk_a === 'string' && <p><strong>Chunk A:</strong> {candidate.chunk_a}</p>}
            {typeof candidate.chunk_b === 'string' && <p><strong>Chunk B:</strong> {candidate.chunk_b}</p>}
          </details>
        ))}
        {unavailable.map((pair, index) => (
          <p className="question-check-warning" key={`unavailable-${index}`}>
            {String(pair.id_a)} ↔ {String(pair.id_b)}: {String(pair.reason)}
          </p>
        ))}
        {candidates.length === 0 && unavailable.length === 0 && <p>No candidate pairs were found in this question’s retrieved records.</p>}
      </div>
    </CheckResultModal>
  );
}

function DriftQuestionResult({ result }: { result?: Record<string, unknown> }) {
  if (!result) return <span className="question-check-muted">Not run</span>;
  const status = String(result.status ?? 'INFO');
  const records = asRecords(result.retrieved_records);
  const differences = Array.isArray(result.differences)
    ? result.differences.filter((difference): difference is string => typeof difference === 'string')
    : [];

  return (
    <CheckResultModal
      title="AUT-Vector Drift"
      summary={
        <>
          <CheckStatus value={status} />
          <span>{typeof result.drift_score === 'number' ? result.drift_score.toFixed(3) : 'No score'}</span>
        </>
      }
    >
      <div className="question-check-detail">
        {typeof result.actual_answer === 'string' && (
          <div className="question-check-answer">
            <strong>Actual AUT answer</strong>
            <p>{result.actual_answer}</p>
          </div>
        )}
        {typeof result.traceback === 'string' && (
          <div className="question-check-answer">
            <strong>{status === 'ERROR' ? 'Error details' : 'Evidence trace'}</strong>
            <p>{result.traceback}</p>
          </div>
        )}
        {differences.length > 0 && (
          <div className="question-check-answer">
            <strong>Material differences</strong>
            <ul>{differences.map((difference, index) => <li key={index}>{difference}</li>)}</ul>
          </div>
        )}
        {records.length > 0 && (
          <details className="question-check-records">
            <summary>Retrieved evidence ({records.length} records)</summary>
            {records.map((record, index) => (
              <div key={`${String(record.id ?? index)}-${index}`}>
                <strong>{String(record.id ?? `Record ${index + 1}`)}</strong>
                <p>{String(record.text ?? '')}</p>
              </div>
            ))}
          </details>
        )}
      </div>
    </CheckResultModal>
  );
}

export default function ExecutionPage() {
  const { activeProject } = useProject();
  const { credentials } = useAzureCredentials();
  const projectId = activeProject?.id ?? '';
  const projectIdRef = useRef(projectId);
  projectIdRef.current = projectId;
  const checksRequestIdRef = useRef(0);
  const restoredSettingsProjectId = useRef('');
  const skipSettingsSaveProjectId = useRef('');
  const [selectedChecks, setSelectedChecks] = useState<string[]>([
    'vector_integrity', 'neighbor_consistency', 'duplicate_detection',
  ]);
  const [requiredFields, setRequiredFields] = useState('pdf_name, pdf_sha256, page_number, chunk_index');
  const [duplicateMin, setDuplicateMin] = useState(0.65);
  const [duplicateMax, setDuplicateMax] = useState(0.98);
  const [selectedNegativeQuestions, setSelectedNegativeQuestions] = useState<string[]>([]);
  const [selectedSuggestedQuestions, setSelectedSuggestedQuestions] = useState<string[]>([]);
  const [offTopicModalOpen, setOffTopicModalOpen] = useState(false);
  const [suggestionModalOpen, setSuggestionModalOpen] = useState(false);
  const [suggestionTopics, setSuggestionTopics] = useState('');
  const [suggestionCount, setSuggestionCount] = useState(5);
  const [suggestions, setSuggestions] = useState<Array<{ topic: string; questions: string[] }>>([]);
  const [negativeThreshold, setNegativeThreshold] = useState(0.32);
  const [report, setReport] = useState<Report | null>(null);
  const [workbook, setWorkbook] = useState<File | null>(null);
  const [autStatus, setAutStatus] = useState<(MockAutStatus & { outdated?: boolean }) | null>(null);
  const [autStatusLoading, setAutStatusLoading] = useState(false);
  const [driftWorkbook, setDriftWorkbook] = useState<NonNullable<WorkbookReport['drift_workbook']> | null>(null);
  const [driftMaxScore, setDriftMaxScore] = useState(0.25);
  const [questionRetrievalTopK, setQuestionRetrievalTopK] = useState(5);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');

  const negativeQuestionList = useMemo(
    () => [...new Set([...selectedNegativeQuestions, ...selectedSuggestedQuestions])],
    [selectedNegativeQuestions, selectedSuggestedQuestions],
  );
  const suggestionTopicList = suggestionTopics
    .split(/\r?\n/)
    .map((topic) => topic.trim())
    .filter(Boolean);
  const canSuggestQuestions = suggestionTopicList.length >= 1
    && suggestionTopicList.length <= 10
    && suggestionTopicList.every((topic) => topic.length <= 120)
    && Number.isInteger(suggestionCount)
    && suggestionCount >= 1
    && suggestionCount <= 10
    && hasAzureCredentials(credentials);

  useEffect(() => {
    if (!projectId) {
      restoredSettingsProjectId.current = '';
      return;
    }
    try {
      const settings = readExecutionSettings(projectId);
      setSelectedChecks(settings?.selectedChecks ?? [
        'vector_integrity', 'neighbor_consistency', 'duplicate_detection',
      ]);
      setRequiredFields(settings?.requiredFields ?? 'pdf_name, pdf_sha256, page_number, chunk_index');
      setDuplicateMin(settings?.duplicateMin ?? 0.65);
      setDuplicateMax(settings?.duplicateMax ?? 0.98);
      setDriftMaxScore(settings?.driftMaxScore ?? 0.25);
      setQuestionRetrievalTopK(settings?.questionRetrievalTopK ?? 5);
      setSelectedNegativeQuestions(settings?.selectedNegativeQuestions ?? []);
      setSelectedSuggestedQuestions(settings?.selectedSuggestedQuestions ?? []);
      setSuggestionTopics(settings?.suggestionTopics ?? '');
      setSuggestionCount(settings?.suggestionCount ?? 5);
      setSuggestions(settings?.suggestions ?? []);
      setNegativeThreshold(settings?.negativeThreshold ?? 0.32);
      restoredSettingsProjectId.current = projectId;
      skipSettingsSaveProjectId.current = projectId;
    } catch (cause) {
      setError(cause instanceof Error
        ? `Could not restore saved check settings: ${cause.message}`
        : 'Could not restore saved check settings.');
      restoredSettingsProjectId.current = projectId;
    }
  }, [projectId]);

  useEffect(() => {
    if (!projectId || restoredSettingsProjectId.current !== projectId) return;
    if (skipSettingsSaveProjectId.current === projectId) {
      skipSettingsSaveProjectId.current = '';
      return;
    }
    const settings: SavedExecutionSettings = {
      selectedChecks,
      requiredFields,
      duplicateMin,
      duplicateMax,
      driftMaxScore,
      questionRetrievalTopK,
      selectedNegativeQuestions,
      selectedSuggestedQuestions,
      suggestionTopics,
      suggestionCount,
      suggestions,
      negativeThreshold,
    };
    try {
      localStorage.setItem(
        `rag_tester_execution_settings_${projectId}`,
        JSON.stringify(settings),
      );
    } catch (cause) {
      setError(cause instanceof Error
        ? `Could not save check settings: ${cause.message}`
        : 'Could not save check settings.');
    }
  }, [
    projectId,
    selectedChecks,
    requiredFields,
    duplicateMin,
    duplicateMax,
    driftMaxScore,
    questionRetrievalTopK,
    selectedNegativeQuestions,
    selectedSuggestedQuestions,
    suggestionTopics,
    suggestionCount,
    suggestions,
    negativeThreshold,
  ]);

  useEffect(() => {
    let current = true;
    checksRequestIdRef.current += 1;
    setBusy('');
    setReport(null);
    setWorkbook(null);
    setDriftWorkbook(null);
    setError('');
    setAutStatus(null);
    setAutStatusLoading(Boolean(projectId));
    if (!projectId) return;
    getMockAutStatus()
      .then((status) => {
        if (current) setAutStatus(status);
      })
      .catch((cause: unknown) => {
        if (current) {
          setError(cause instanceof Error ? cause.message : 'Could not load Mock AUT status.');
        }
      })
      .finally(() => {
        if (current) setAutStatusLoading(false);
      });
    return () => {
      current = false;
    };
  }, [projectId]);

  useEffect(() => {
    if (!offTopicModalOpen && !suggestionModalOpen) return undefined;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setOffTopicModalOpen(false);
        setSuggestionModalOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [offTopicModalOpen, suggestionModalOpen]);

  const toggleCheck = (check: string) => {
    setSelectedChecks((current) => current.includes(check)
      ? current.filter((value) => value !== check)
      : [...current, check]);
    setReport(null);
  };

  const suggestQuestions = async () => {
    setBusy('suggestions');
    setError('');
    try {
      const result = await suggestOffTopicQuestions(
        projectId,
        suggestionTopicList,
        suggestionCount,
        credentials,
      );
      setSuggestions(result.suggestions);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not suggest off-topic questions.');
    } finally {
      setBusy('');
    }
  };

  const runChecks = async () => {
    const requestId = ++checksRequestIdRef.current;
    const requestProjectId = projectId;
    setBusy('checks');
    setError('');
    try {
      const config = {
        selected_checks: selectedChecks,
        required_fields: requiredFields.split(',').map((field) => field.trim()).filter(Boolean),
        duplicate_min_similarity: duplicateMin,
        duplicate_max_similarity: duplicateMax,
        negative_query_texts: negativeQuestionList,
        negative_query_max_similarity: negativeThreshold,
        drift_max_score: driftMaxScore,
        question_retrieval_top_k: questionRetrievalTopK,
      };
      let workbookCheckResult: WorkbookReport | null = null;
      const result = workbook
        ? (workbookCheckResult = await runProjectWorkbookChecks(
          projectId,
          workbook,
          config,
          credentials,
        ))
        : await runProjectVectorChecks(projectId, config);
      if (
        requestId !== checksRequestIdRef.current
        || projectIdRef.current !== requestProjectId
      ) {
        return;
      }
      setReport(result);
      setDriftWorkbook(workbookCheckResult?.drift_workbook ?? null);
    } catch (cause) {
      if (
        requestId === checksRequestIdRef.current
        && projectIdRef.current === requestProjectId
      ) {
        setError(cause instanceof Error ? cause.message : 'Vector checks failed.');
      }
    } finally {
      if (
        requestId === checksRequestIdRef.current
        && projectIdRef.current === requestProjectId
      ) {
        setBusy('');
      }
    }
  };

  const workbookQuestionsRequired = selectedChecks.includes('neighbor_consistency')
    || selectedChecks.includes('aut_vector_drift')
    || selectedChecks.includes('duplicate_detection')
    || selectedChecks.includes('hit_at_k');
  const mockAutReady = Boolean(autStatus?.running && !autStatus.outdated);

  const invalidConfiguration = !selectedChecks.length
    || (selectedChecks.includes('duplicate_detection') && (
      duplicateMin < 0.65 || duplicateMax > 0.98 || duplicateMin > duplicateMax
    ))
    || (selectedChecks.includes('negative_query_similarity') && (
      !negativeQuestionList.length
      || negativeQuestionList.length > 100
      || negativeQuestionList.some((question) => question.length > 2000)
    ))
    || (workbookQuestionsRequired && !workbook)
    || ((selectedChecks.includes('neighbor_consistency')
      || selectedChecks.includes('aut_vector_drift')
      || selectedChecks.includes('duplicate_detection'))
      && (!Number.isInteger(questionRetrievalTopK) || questionRetrievalTopK < 1 || questionRetrievalTopK > 20))
    || (selectedChecks.includes('aut_vector_drift') && (
      !hasAzureCredentials(credentials)
      || (!autStatusLoading && autStatus !== null && !mockAutReady)
      || driftMaxScore < 0
      || driftMaxScore > 1
    ));

  return (
    <div className="space-y-6">
      <header className="page-heading">
        <div>
          <p className="eyebrow">Project assurance</p>
          <h1>Execution</h1>
          <p>Run quality checks against the selected vector database for {activeProject?.name ?? 'the active project'}.</p>
        </div>
      </header>

      {error && <div className="alert-error flex items-start gap-2"><AlertCircle size={17} />{error}</div>}

      <section className="panel space-y-4">
        <div className="section-title">
          <span className="icon-tile"><FileUp size={19} /></span>
          <div><h2>Q&amp;A workbook</h2><p>Upload the workbook from Test Data Generation. Its questions are available to workbook-based checks below.</p></div>
        </div>
        <label className="field-label">
          Questions workbook (.xlsx)
          <input
            type="file"
            accept=".xlsx,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            onChange={(event) => {
              setWorkbook(event.target.files?.[0] ?? null);
              setReport(null);
              setDriftWorkbook(null);
            }}
          />
          <span className="field-help">The workbook must contain a question column. Hit@k also requires object_type, chunk, and object_image_paths columns. Neighbor Consistency and Duplicate Detection analyze each question’s retrieved records; AUT-Vector Drift judges the AUT answer against that question’s evidence.</span>
        </label>
        {workbook && <p className="text-xs text-slate-500">{workbook.name}</p>}
        {selectedChecks.includes('aut_vector_drift') && autStatusLoading && (
          <p className="field-help">Checking Mock AUT status. The backend will confirm availability when checks start.</p>
        )}
        {selectedChecks.includes('aut_vector_drift') && !autStatusLoading && !mockAutReady && (
          <p className="field-help">Start or restart the Mock AUT in <Link className="text-blue-700 underline" to="/app-under-test">Configure</Link> before running AUT-Vector Drift.</p>
        )}
        {selectedChecks.includes('aut_vector_drift') && !hasAzureCredentials(credentials) && (
          <p className="field-help">Configure Azure OpenAI credentials in <Link className="text-blue-700 underline" to="/app-under-test">Configure</Link> before running AUT-Vector Drift.</p>
        )}
        {driftWorkbook && (
          <div className="alert-success flex flex-wrap items-center justify-between gap-3">
            <span>AUT-Vector Drift workbook is ready with per-question answers, retrieved records, scores, and drift traces.</span>
            <button
              className="button-secondary"
              onClick={() => downloadBase64File(
                driftWorkbook.filename,
                driftWorkbook.content_base64,
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
              )}
            >Download drift workbook</button>
          </div>
        )}
      </section>

      <section className="panel space-y-4">
        <div className="section-title">
          <span className="icon-tile"><Play size={19} /></span>
          <div><h2>Metric Selection</h2><p>Choose checks and configure any required thresholds.</p></div>
        </div>
        <div className="grid items-start gap-3 md:grid-cols-2">
          {CHECKS.map((check) => {
            const checked = selectedChecks.includes(check.id);
            return (
              <article
                key={check.id}
                className={`execution-check-card ${checked ? 'execution-check-card-selected' : ''}`}
              >
                <label className="execution-check-heading">
                  <input type="checkbox" checked={checked} onChange={() => toggleCheck(check.id)} />
                  <span>
                    <strong>{check.label}</strong>
                    <span>{check.description}</span>
                  </span>
                </label>
                {checked && check.id === 'metadata_completeness' && (
                  <div className="execution-check-options">
                    <label className="field-label">Required metadata fields
                      <input value={requiredFields} onChange={(event) => setRequiredFields(event.target.value)} />
                      <span className="field-help">Separate field names with commas.</span>
                    </label>
                  </div>
                )}
                {checked && check.id === 'neighbor_consistency' && (
                  <div className="execution-check-options">
                    <label className="field-label">Records retrieved per workbook question (top-k)
                      <input
                        type="number"
                        min={1}
                        max={20}
                        step={1}
                        value={questionRetrievalTopK}
                        onChange={(event) => setQuestionRetrievalTopK(Number(event.target.value))}
                      />
                      <span className="field-help">Each question is retrieved and checked separately. Every distinct pair among that question’s retrieved records is checked.</span>
                    </label>
                  </div>
                )}
                {checked && check.id === 'aut_vector_drift' && (
                  <div className="execution-check-options grid gap-3 sm:grid-cols-2">
                    <label className="field-label">Maximum allowed drift (0–1)
                      <input
                        type="number"
                        min={0}
                        max={1}
                        step={0.05}
                        value={driftMaxScore}
                        onChange={(event) => setDriftMaxScore(Number(event.target.value))}
                      />
                      <span className="field-help">0 means answers agree; 1 means they materially diverge.</span>
                    </label>
                    {selectedChecks.includes('neighbor_consistency') ? (
                      <p className="field-help sm:col-span-2">AUT-Vector Drift uses the same top-k retrieval setting shown under Neighbor Consistency.</p>
                    ) : (
                      <label className="field-label">Records retrieved per question (top-k)
                        <input
                          type="number"
                          min={1}
                          max={20}
                          step={1}
                          value={questionRetrievalTopK}
                          onChange={(event) => setQuestionRetrievalTopK(Number(event.target.value))}
                        />
                      </label>
                    )}
                  </div>
                )}
                {checked && check.id === 'duplicate_detection' && (
                  <div className="execution-check-options grid gap-3 sm:grid-cols-2">
                    <label className="field-label">Similarity from
                      <input type="number" min={0.65} max={0.98} step={0.01} value={duplicateMin} onChange={(event) => setDuplicateMin(Number(event.target.value))} />
                    </label>
                    <label className="field-label">Similarity to
                      <input type="number" min={0.65} max={0.98} step={0.01} value={duplicateMax} onChange={(event) => setDuplicateMax(Number(event.target.value))} />
                    </label>
                    {!selectedChecks.includes('neighbor_consistency')
                      && !selectedChecks.includes('aut_vector_drift') && (
                        <label className="field-label">Records retrieved per workbook question (top-k)
                          <input
                            type="number"
                            min={1}
                            max={20}
                            step={1}
                            value={questionRetrievalTopK}
                            onChange={(event) => setQuestionRetrievalTopK(Number(event.target.value))}
                          />
                          <span className="field-help">Duplicate candidates are compared among each question’s retrieved records.</span>
                        </label>
                      )}
                  </div>
                )}
                {checked && check.id === 'negative_query_similarity' && (
                  <div className="execution-check-options grid gap-3">
                    <div className="grid gap-2 sm:grid-cols-2">
                      <button
                        className="button-secondary execution-negative-action"
                        type="button"
                        onClick={() => setOffTopicModalOpen(true)}
                      >
                        <ListChecks size={17} />
                        Off-topic questions
                        <span className="object-type-count">{selectedNegativeQuestions.length} selected</span>
                      </button>
                      <button
                        className="button-secondary execution-negative-action"
                        type="button"
                        onClick={() => setSuggestionModalOpen(true)}
                      >
                        <Sparkles size={17} />
                        Suggest questions
                        <span className="object-type-count">{selectedSuggestedQuestions.length} selected</span>
                      </button>
                    </div>
                    <p className="field-help">
                      {negativeQuestionList.length} total question{negativeQuestionList.length === 1 ? '' : 's'} selected; maximum 100.
                    </p>
                    <label className="field-label">Maximum cosine similarity
                      <input type="number" min={0} max={1} step={0.01} value={negativeThreshold} onChange={(event) => setNegativeThreshold(Number(event.target.value))} />
                    </label>
                  </div>
                )}
              </article>
            );
          })}
        </div>

        <button className="button-primary" disabled={invalidConfiguration || Boolean(busy)} onClick={runChecks}>
          {busy === 'checks' ? <><RefreshCw size={15} className="animate-spin" />Running checks…</> : 'Run selected checks'}
        </button>
      </section>

      {offTopicModalOpen && (
        <div
          className="execution-negative-modal-backdrop"
          onClick={() => setOffTopicModalOpen(false)}
        >
          <section
            className="execution-negative-modal execution-negative-modal-wide"
            role="dialog"
            aria-modal="true"
            aria-labelledby="off-topic-title"
            onClick={(event) => event.stopPropagation()}
          >
            <header className="execution-negative-modal-header">
              <div>
                <p className="eyebrow">Off-topic questions</p>
                <h2 id="off-topic-title">Choose questions by topic</h2>
                <p>Expand a topic and select the questions to include in this check.</p>
              </div>
              <button
                className="object-preview-close"
                type="button"
                aria-label="Close off-topic questions"
                onClick={() => setOffTopicModalOpen(false)}
              >
                <X size={20} />
              </button>
            </header>
            <div className="execution-negative-category-list">
              {OFF_TOPIC_CATEGORIES.map((category) => {
                const selectedCount = category.questions.filter(
                  (question) => selectedNegativeQuestions.includes(question),
                ).length;
                return (
                  <details className="execution-negative-topic" key={category.id}>
                    <summary>
                      <span>{category.label}</span>
                      <span className="object-type-count">{selectedCount} selected</span>
                    </summary>
                    <div className="execution-negative-question-list">
                      {category.questions.map((question) => {
                        const selected = selectedNegativeQuestions.includes(question);
                        return (
                          <label key={question}>
                            <input
                              type="checkbox"
                              checked={selected}
                              disabled={!selected && negativeQuestionList.length >= 100}
                              onChange={(event) => {
                                setSelectedNegativeQuestions((current) => event.target.checked
                                  ? [...current, question]
                                  : current.filter((value) => value !== question));
                              }}
                            />
                            <span>{question}</span>
                          </label>
                        );
                      })}
                    </div>
                  </details>
                );
              })}
            </div>
            <footer className="execution-negative-modal-footer">
              <span>{negativeQuestionList.length} total question{negativeQuestionList.length === 1 ? '' : 's'} selected</span>
              <button
                className="button-primary"
                type="button"
                onClick={() => setOffTopicModalOpen(false)}
              >
                Done
              </button>
            </footer>
          </section>
        </div>
      )}

      {suggestionModalOpen && (
        <div
          className="execution-negative-modal-backdrop"
          onClick={() => setSuggestionModalOpen(false)}
        >
          <section
            className="execution-negative-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="suggest-off-topic-title"
            onClick={(event) => event.stopPropagation()}
          >
            <header className="execution-negative-modal-header">
              <div>
                <p className="eyebrow">AI question suggestions</p>
                <h2 id="suggest-off-topic-title">Suggest questions by topic</h2>
                <p>Enter one or more topics. Questions are generated separately for each topic.</p>
              </div>
              <button
                className="object-preview-close"
                type="button"
                aria-label="Close question suggestions"
                onClick={() => setSuggestionModalOpen(false)}
              >
                <X size={20} />
              </button>
            </header>
            <div className="execution-suggestion-body">
              <label className="field-label">Topics (one per line)
                <textarea
                  rows={3}
                  maxLength={1200}
                  value={suggestionTopics}
                  onChange={(event) => setSuggestionTopics(event.target.value)}
                  placeholder={'Insurance claims\nPersonal finance\nHome renovation'}
                />
                <span className="field-help">Enter 1–10 topics; each topic can be up to 120 characters.</span>
              </label>
              <label className="field-label">Questions per topic
                <input
                  type="number"
                  min={1}
                  max={10}
                  step={1}
                  value={suggestionCount}
                  onChange={(event) => setSuggestionCount(Number(event.target.value))}
                />
              </label>
              {!hasAzureCredentials(credentials) && (
                <p className="field-help">
                  Configure Azure OpenAI credentials in <Link className="text-blue-700 underline" to="/app-under-test">Configure</Link> to suggest questions.
                </p>
              )}
              <button
                className="button-primary"
                type="button"
                disabled={!projectId || !canSuggestQuestions || Boolean(busy)}
                onClick={suggestQuestions}
              >
                {busy === 'suggestions'
                  ? <><RefreshCw size={15} className="animate-spin" />Generating…</>
                  : <><Sparkles size={16} />Generate questions</>}
              </button>

              {error && busy !== 'suggestions' && (
                <div className="alert-error" role="alert">{error}</div>
              )}
              {suggestions.length > 0 && (
                <div className="execution-negative-suggestions">
                  <h3>Choose questions to include</h3>
                  {suggestions.map((suggestion) => (
                    <fieldset className="execution-negative-categories" key={suggestion.topic}>
                      <legend>{suggestion.topic}</legend>
                      <div className="execution-negative-question-list">
                        {suggestion.questions.map((question) => {
                          const selected = selectedSuggestedQuestions.includes(question);
                          return (
                            <label key={question}>
                              <input
                                type="checkbox"
                                checked={selected}
                                disabled={!selected && negativeQuestionList.length >= 100}
                                onChange={(event) => setSelectedSuggestedQuestions((current) => event.target.checked
                                  ? [...current, question]
                                  : current.filter((value) => value !== question))}
                              />
                              <span>{question}</span>
                            </label>
                          );
                        })}
                      </div>
                    </fieldset>
                  ))}
                </div>
              )}
            </div>
            <footer className="execution-negative-modal-footer">
              <span>{negativeQuestionList.length} total question{negativeQuestionList.length === 1 ? '' : 's'} selected</span>
              <button
                className="button-primary"
                type="button"
                onClick={() => setSuggestionModalOpen(false)}
              >
                Done
              </button>
            </footer>
          </section>
        </div>
      )}

      {report && (
        <section className="panel execution-results-panel">
          <div className="execution-results-header">
            <div className="section-title">
              <span className="icon-tile"><ListChecks size={18} /></span>
              <div>
                <h2>Run results</h2>
                <p>Report {report.report_id} · completed check summary</p>
              </div>
            </div>
            <button className="button-secondary" onClick={() => downloadHtml(report.html_report, report.report_id)}><Download size={15} />Download HTML report</button>
          </div>
          <div className="metric-grid execution-results-summary">
            <div className="metric-card execution-summary-card execution-summary-passed"><strong>{report.summary.passed}</strong><span>Passed</span></div>
            <div className="metric-card execution-summary-card execution-summary-failed"><strong>{report.summary.failed}</strong><span>Failed</span></div>
            <div className="metric-card execution-summary-card execution-summary-info"><strong>{report.summary.inconclusive}</strong><span>Informational</span></div>
            <div className="metric-card execution-summary-card execution-summary-total"><strong>{report.summary.total}</strong><span>Total checks</span></div>
          </div>
          <QuestionWiseResults report={report} selectedChecks={selectedChecks} />
          {report.results
            .filter((result) => result.check === 'hit_at_k')
            .map((result, index) => (
              <HitAtKResults key={`${result.check}-${index}`} result={result} />
            ))}
          {report.results.some((result) => ![
            'known_similar_pairs',
            'near_duplicate_detection_by_question',
            'AUT-Vector Drift',
            'dimension_consistency_by_question',
            'metadata_completeness_by_question',
            'hit_at_k',
          ].includes(result.check)) && (
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead><tr><th>Check</th><th>Status</th></tr></thead>
                <tbody>{report.results
                  .filter((result) => ![
                    'known_similar_pairs',
                    'near_duplicate_detection_by_question',
                    'AUT-Vector Drift',
                    'dimension_consistency_by_question',
                    'metadata_completeness_by_question',
                    'hit_at_k',
                  ].includes(result.check))
                  .map((result, index) => (
                    <tr key={`${result.check}-${index}`}>
                      <td className="execution-result-name">
                        <CheckResultModal
                          title={result.check}
                          summary={<span className="execution-result-open">{result.check}</span>}
                        >
                          <pre className="execution-result-json">{JSON.stringify(result.detail, null, 2)}</pre>
                        </CheckResultModal>
                      </td>
                      <td><CheckStatus value={result.passed === true ? 'PASS' : result.passed === false ? 'FAIL' : 'INFO'} /></td>
                    </tr>
                  ))}</tbody>
              </table>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
