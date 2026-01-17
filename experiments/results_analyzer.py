import os
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, List, Any, Optional
from datetime import datetime
import glob


class ResultsAnalyzer:
    """Analyzer for individual results files."""
    
    def __init__(self, results_path: str):
        self.results_path = results_path
        self.data: Dict[str, Any] = {}
        self.results: List[Dict] = []
        self.experiment_type = self._detect_experiment_type()
        self.load_results()
    
    def _detect_experiment_type(self) -> str:
        """Detect which experiment type based on file path or content."""
        path_lower = self.results_path.lower()
        if 'exp1' in path_lower or ('results_k' in path_lower and 'oracle' not in path_lower 
                                     and 'noise' not in path_lower and 'hardneg' not in path_lower
                                     and 'exp5' not in path_lower):
            return 'exp1'
        elif 'exp2' in path_lower or 'oracle' in path_lower:
            return 'exp2'
        elif 'exp3' in path_lower or 'noise' in path_lower:
            return 'exp3'
        elif 'exp4' in path_lower or 'hardneg' in path_lower:
            return 'exp4'
        elif 'exp5' in path_lower:
            return 'exp5'
        return 'unknown'
    
    def load_results(self):
        with open(self.results_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.results = self.data.get('results', [])
    
    def get_basic_stats(self) -> Dict[str, Any]:
        stats = {
            'experiment_type': self.experiment_type,
            'k': self.data.get('k', 'N/A'),
            'total_questions': self.data.get('total', len(self.results)),
            'matches': self.data.get('matches', sum(1 for r in self.results if r.get('is_match', False))),
            'exact_match': self.data.get('exact_match', 0),
            'errors': self.data.get('errors', 0),
            'timestamp': self.data.get('timestamp', 'N/A')
        }
        
        # Experiment-specific fields
        if self.experiment_type in ['exp3', 'exp4']:
            stats['total_docs'] = self.data.get('total_docs', 'N/A')
            stats['num_noise'] = self.data.get('num_noise', self.data.get('num_hard_negs', 'N/A'))
        
        if self.experiment_type == 'exp2':
            stats['avg_oracle_contexts'] = self._compute_avg_oracle_contexts()
        
        # Gold retrieval stats (exp1, exp3, exp4, exp5)
        if 'gold_recall_rate' in self.data:
            stats['gold_recall_rate'] = self.data.get('gold_recall_rate', 0)
            stats['gold_em_when_retrieved'] = self.data.get('gold_em_when_retrieved', 0)
            stats['all_gold_retrieved_count'] = self.data.get('all_gold_retrieved_count', 0)
        
        return stats
    
    def _compute_avg_oracle_contexts(self) -> float:
        if not self.results:
            return 0.0
        total = sum(r.get('num_oracle_contexts', 0) for r in self.results)
        return total / len(self.results)
    
    def analyze_gold_docs(self) -> Dict[str, Any]:
        """Analyze golden document retrieval statistics."""
        matches = [r for r in self.results if r.get('is_match', False)]
        non_matches = [r for r in self.results if not r.get('is_match', False)]
        
        # Different field names for different experiments
        gold_field = 'gold_docs_in_topk'
        all_gold_field = 'all_gold_in_topk'
        
        if self.experiment_type in ['exp3', 'exp4']:
            gold_field = 'gold_in_top_k'
            all_gold_field = 'all_gold_retrieved'
        
        matches_with_gold = 0
        matches_without_gold = 0
        matches_with_all_gold = 0
        
        for r in matches:
            gold_in_topk = r.get(gold_field, 0)
            all_gold = r.get(all_gold_field, False)
            
            if gold_in_topk >= 1:
                matches_with_gold += 1
            else:
                matches_without_gold += 1
            
            if all_gold:
                matches_with_all_gold += 1
        
        non_matches_with_gold = 0
        non_matches_with_all_gold = 0
        
        for r in non_matches:
            gold_in_topk = r.get(gold_field, 0)
            all_gold = r.get(all_gold_field, False)
            
            if gold_in_topk >= 1:
                non_matches_with_gold += 1
            if all_gold:
                non_matches_with_all_gold += 1
        
        total_matches = len(matches)
        total_non_matches = len(non_matches)
        
        return {
            'total_matches': total_matches,
            'total_non_matches': total_non_matches,
            'matches_with_at_least_1_gold': matches_with_gold,
            'matches_without_any_gold': matches_without_gold,
            'matches_with_all_gold': matches_with_all_gold,
            'non_matches_with_at_least_1_gold': non_matches_with_gold,
            'non_matches_with_all_gold': non_matches_with_all_gold,
            'fraction_matches_with_gold': matches_with_gold / total_matches if total_matches > 0 else 0,
            'fraction_matches_with_all_gold': matches_with_all_gold / total_matches if total_matches > 0 else 0,
            'fraction_non_matches_with_gold': non_matches_with_gold / total_non_matches if total_non_matches > 0 else 0,
        }
    
    def get_gold_distribution(self) -> Dict[int, int]:
        """Get distribution of gold docs in top-k."""
        distribution = {}
        
        gold_field = 'gold_docs_in_topk'
        if self.experiment_type in ['exp3', 'exp4']:
            gold_field = 'gold_in_top_k'
        
        for r in self.results:
            gold_in_topk = r.get(gold_field, 0)
            distribution[gold_in_topk] = distribution.get(gold_in_topk, 0) + 1
        return dict(sorted(distribution.items()))
    
    def generate_analysis_text(self) -> str:
        """Generate full analysis as text (for file output)."""
        lines = []
        
        basic = self.get_basic_stats()
        gold = self.analyze_gold_docs()
        dist = self.get_gold_distribution()
        
        lines.append("=" * 60)
        lines.append("BASIC STATISTICS")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"File: {os.path.basename(self.results_path)}")
        lines.append(f"Experiment Type: {basic['experiment_type'].upper()}")
        lines.append(f"k value: {basic['k']}")
        
        if 'total_docs' in basic:
            lines.append(f"Total docs per query: {basic['total_docs']}")
            lines.append(f"Noise/Hard-neg docs: {basic['num_noise']}")
        
        lines.append(f"Total questions: {basic['total_questions']}")
        lines.append(f"Correct answers (matches): {basic['matches']}")
        lines.append(f"Exact Match score: {basic['exact_match']:.4f} ({basic['exact_match']*100:.2f}%)")
        lines.append(f"Errors: {basic['errors']}")
        
        if 'gold_recall_rate' in basic:
            lines.append(f"Gold Recall Rate: {basic['gold_recall_rate']:.4f} ({basic['gold_recall_rate']*100:.2f}%)")
            lines.append(f"Gold EM When Retrieved: {basic['gold_em_when_retrieved']:.4f} ({basic['gold_em_when_retrieved']*100:.2f}%)")
        
        if 'avg_oracle_contexts' in basic:
            lines.append(f"Avg Oracle Contexts: {basic['avg_oracle_contexts']:.3f}")
        
        lines.append(f"Timestamp: {basic['timestamp']}")
        lines.append("")
        
        # Skip gold doc analysis for exp2 (oracle - no retrieval)
        if basic['experiment_type'] != 'exp2':
            lines.append("=" * 60)
            lines.append("GOLDEN DOCUMENT ANALYSIS")
            lines.append("=" * 60)
            lines.append("")
            
            lines.append("For CORRECT answers (matches):")
            lines.append("-" * 40)
            lines.append(f"Total matches: {gold['total_matches']}")
            lines.append(f"Matches with >= 1 gold doc in retrieved: {gold['matches_with_at_least_1_gold']}")
            lines.append(f"Matches without any gold doc: {gold['matches_without_any_gold']}")
            lines.append(f"Matches with ALL gold docs in retrieved: {gold['matches_with_all_gold']}")
            lines.append("")
            
            frac = gold['fraction_matches_with_gold']
            lines.append(f">>> Fraction of matches with at least 1 gold doc: {frac:.4f} ({frac*100:.2f}%)")
            
            frac_all = gold['fraction_matches_with_all_gold']
            lines.append(f">>> Fraction of matches with ALL gold docs: {frac_all:.4f} ({frac_all*100:.2f}%)")
            lines.append("")
            
            lines.append("For INCORRECT answers (non-matches):")
            lines.append("-" * 40)
            lines.append(f"Total non-matches: {gold['total_non_matches']}")
            lines.append(f"Non-matches with >= 1 gold doc: {gold['non_matches_with_at_least_1_gold']}")
            lines.append(f"Non-matches with ALL gold docs: {gold['non_matches_with_all_gold']}")
            
            frac_nm = gold['fraction_non_matches_with_gold']
            lines.append(f">>> Fraction of non-matches with at least 1 gold doc: {frac_nm:.4f} ({frac_nm*100:.2f}%)")
            lines.append("")
            
            lines.append("=" * 60)
            lines.append("GOLD DOCS IN TOP-K DISTRIBUTION")
            lines.append("=" * 60)
            lines.append("")
            
            for num_gold, count in dist.items():
                pct = count / len(self.results) * 100
                bar = "#" * int(pct / 2)
                lines.append(f"{num_gold} gold doc(s): {count:4d} ({pct:5.1f}%) {bar}")
        
        return "\n".join(lines)


class SummaryAnalyzer:
    """Analyzer for experiment summary files."""
    
    def __init__(self, summary_path: str):
        self.summary_path = summary_path
        self.data: Dict[str, Any] = {}
        self.experiment_type = self._detect_experiment_type()
        self.load_summary()
    
    def _detect_experiment_type(self) -> str:
        path_lower = self.summary_path.lower()
        if 'experiment1' in path_lower:
            return 'exp1'
        elif 'experiment2' in path_lower or 'oracle' in path_lower:
            return 'exp2'
        elif 'experiment3' in path_lower or 'noise' in path_lower:
            return 'exp3'
        elif 'experiment4' in path_lower or 'hardneg' in path_lower:
            return 'exp4'
        elif 'experiment5' in path_lower:
            return 'exp5'
        return 'unknown'
    
    def load_summary(self):
        with open(self.summary_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def generate_analysis_text(self) -> str:
        """Generate analysis text from summary."""
        lines = []
        
        lines.append("=" * 60)
        lines.append("EXPERIMENT SUMMARY ANALYSIS")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"File: {os.path.basename(self.summary_path)}")
        lines.append(f"Experiment: {self.data.get('experiment', 'N/A')}")
        
        if 'description' in self.data:
            lines.append(f"Description: {self.data.get('description', '')}")
        
        lines.append(f"Number of Questions: {self.data.get('num_questions', 'N/A')}")
        
        if 'retriever' in self.data:
            lines.append(f"Retriever: {self.data.get('retriever', 'N/A')}")
        if 'llm' in self.data:
            lines.append(f"LLM: {self.data.get('llm', self.data.get('model', 'N/A'))}")
        
        lines.append(f"K Values: {self.data.get('k_values', 'N/A')}")
        
        if 'total_docs_values' in self.data:
            lines.append(f"Total Docs Values: {self.data.get('total_docs_values', 'N/A')}")
        
        if 'avg_oracle_contexts' in self.data:
            lines.append(f"Avg Oracle Contexts: {self.data.get('avg_oracle_contexts', 'N/A')}")
        
        lines.append(f"Quick Mode: {self.data.get('quick_mode', 'N/A')}")
        lines.append(f"Timestamp: {self.data.get('timestamp', 'N/A')}")
        lines.append("")
        
        # Results breakdown
        results = self.data.get('results', {})
        if results:
            lines.append("=" * 60)
            lines.append("RESULTS BY CONFIGURATION")
            lines.append("=" * 60)
            lines.append("")
            
            for config_key, config_data in results.items():
                lines.append(f"--- Configuration: {config_key} ---")
                
                if isinstance(config_data, dict):
                    em = config_data.get('exact_match', config_data.get('em', 0))
                    lines.append(f"  Exact Match: {em:.4f} ({em*100:.2f}%)")
                    
                    if 'matches' in config_data:
                        lines.append(f"  Matches: {config_data.get('matches', 'N/A')}")
                    if 'total' in config_data:
                        lines.append(f"  Total: {config_data.get('total', 'N/A')}")
                    
                    if 'gold_recall_rate' in config_data:
                        gr = config_data.get('gold_recall_rate', 0)
                        lines.append(f"  Gold Recall Rate: {gr:.4f} ({gr*100:.2f}%)")
                    
                    if 'gold_em_when_retrieved' in config_data:
                        gem = config_data.get('gold_em_when_retrieved', 0)
                        lines.append(f"  Gold EM When Retrieved: {gem:.4f} ({gem*100:.2f}%)")
                    
                    if 'all_gold_retrieved_count' in config_data:
                        lines.append(f"  All Gold Retrieved Count: {config_data.get('all_gold_retrieved_count', 'N/A')}")
                
                lines.append("")
        
        return "\n".join(lines)


class BatchAnalyzer:
    """Batch analyzer for all experiment folders."""
    
    EXPERIMENT_PATTERNS = {
        'exp1': {
            'results': ['results_k*.json'],
            'summary': ['experiment1_summary.json']
        },
        'exp2': {
            'results': ['oracle_results_k*.json'],
            'summary': ['experiment2_oracle_summary.json']
        },
        'exp3': {
            'results': ['noise_results_k*_total*.json'],
            'summary': ['experiment3_noise_summary.json']
        },
        'exp4': {
            'results': ['hardneg_results_k*_total*.json'],
            'summary': ['experiment4_hardneg_summary.json']
        },
        'exp5': {
            'results': ['exp5_results_k*.json'],
            'summary': ['experiment5_summary.json', 'experiment5_summary_quick.json']
        }
    }
    
    def __init__(self, base_results_dir: str):
        self.base_results_dir = base_results_dir
        self.output_dir = os.path.join(base_results_dir, 'analyzed_results')
    
    def run_batch_analysis(self, progress_callback=None) -> Dict[str, List[str]]:
        """Run analysis on all experiment folders."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        analyzed_files = {'results': [], 'summaries': []}
        total_files = 0
        processed = 0
        
        # Count total files first
        for exp_num in range(1, 6):
            exp_folder = f'exp{exp_num}'
            exp_path = os.path.join(self.base_results_dir, exp_folder)
            if os.path.exists(exp_path):
                patterns = self.EXPERIMENT_PATTERNS.get(exp_folder, {})
                for pattern in patterns.get('results', []):
                    total_files += len(glob.glob(os.path.join(exp_path, pattern)))
                for pattern in patterns.get('summary', []):
                    total_files += len(glob.glob(os.path.join(exp_path, pattern)))
        
        # Process each experiment folder
        for exp_num in range(1, 6):
            exp_folder = f'exp{exp_num}'
            exp_path = os.path.join(self.base_results_dir, exp_folder)
            
            if not os.path.exists(exp_path):
                continue
            
            # Create output subfolder
            exp_output_dir = os.path.join(self.output_dir, exp_folder)
            os.makedirs(exp_output_dir, exist_ok=True)
            
            patterns = self.EXPERIMENT_PATTERNS.get(exp_folder, {})
            
            # Process results files
            for pattern in patterns.get('results', []):
                for results_file in glob.glob(os.path.join(exp_path, pattern)):
                    try:
                        analyzer = ResultsAnalyzer(results_file)
                        analysis_text = analyzer.generate_analysis_text()
                        
                        # Save analysis
                        base_name = os.path.splitext(os.path.basename(results_file))[0]
                        output_file = os.path.join(exp_output_dir, f"{base_name}_analysis.txt")
                        
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(analysis_text)
                        
                        analyzed_files['results'].append(output_file)
                        processed += 1
                        
                        if progress_callback:
                            progress_callback(processed, total_files, f"Analyzed: {os.path.basename(results_file)}")
                    
                    except Exception as e:
                        print(f"Error analyzing {results_file}: {e}")
            
            # Process summary files
            for pattern in patterns.get('summary', []):
                for summary_file in glob.glob(os.path.join(exp_path, pattern)):
                    try:
                        analyzer = SummaryAnalyzer(summary_file)
                        analysis_text = analyzer.generate_analysis_text()
                        
                        base_name = os.path.splitext(os.path.basename(summary_file))[0]
                        output_file = os.path.join(exp_output_dir, f"{base_name}_analysis.txt")
                        
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(analysis_text)
                        
                        analyzed_files['summaries'].append(output_file)
                        processed += 1
                        
                        if progress_callback:
                            progress_callback(processed, total_files, f"Analyzed: {os.path.basename(summary_file)}")
                    
                    except Exception as e:
                        print(f"Error analyzing {summary_file}: {e}")
        
        return analyzed_files


class ResultsAnalyzerUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Experiment Results Analyzer")
        self.root.geometry("900x800")
        self.root.minsize(800, 700)
        
        self.analyzer: Optional[ResultsAnalyzer] = None
        self.current_file = tk.StringVar(value="No file selected")
        
        self._create_widgets()
    
    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title = ttk.Label(main_frame, text="Experiment Results Analyzer", 
                         font=('Helvetica', 16, 'bold'))
        title.pack(pady=(0, 15))
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Tab 1: Single File Analysis
        self.single_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.single_tab, text="Single File Analysis")
        self._create_single_file_tab()
        
        # Tab 2: Batch Analysis
        self.batch_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.batch_tab, text="Batch Analysis (All Experiments)")
        self._create_batch_tab()
    
    def _create_single_file_tab(self):
        # File selection frame
        file_frame = ttk.LabelFrame(self.single_tab, text="Select Results File", padding=10)
        file_frame.pack(fill='x', pady=5)
        
        self.file_label = ttk.Label(file_frame, textvariable=self.current_file, 
                                    font=('Consolas', 9), foreground='gray')
        self.file_label.pack(fill='x', pady=5)
        
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill='x')
        
        ttk.Button(btn_frame, text="Browse...", command=self._browse_file).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Analyze", command=self._analyze).pack(side='left', padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.single_tab, text="Analysis Results", padding=10)
        results_frame.pack(fill='both', expand=True, pady=10)
        
        # Text widget for results
        self.results_text = tk.Text(results_frame, font=('Consolas', 10), 
                                    wrap='word', state='disabled', height=25)
        self.results_text.pack(side='left', fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(results_frame, orient='vertical', 
                                  command=self.results_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.results_text.config(yscrollcommand=scrollbar.set)
        
        # Configure tags for colored text
        self.results_text.tag_configure('header', font=('Consolas', 11, 'bold'))
        self.results_text.tag_configure('highlight', foreground='#006600', font=('Consolas', 10, 'bold'))
        self.results_text.tag_configure('warning', foreground='#CC6600')
        self.results_text.tag_configure('metric', foreground='#0066CC')
    
    def _create_batch_tab(self):
        # Info label
        info_label = ttk.Label(self.batch_tab, 
                              text="Batch analyze all results files in exp1-exp5 folders.\n"
                                   "Output will be saved to 'analyzed_results' folder.",
                              font=('Helvetica', 10))
        info_label.pack(pady=10)
        
        # Base directory selection
        dir_frame = ttk.LabelFrame(self.batch_tab, text="Results Directory", padding=10)
        dir_frame.pack(fill='x', pady=5)
        
        self.batch_dir = tk.StringVar(value=os.path.join(os.path.dirname(__file__), 'results'))
        
        ttk.Label(dir_frame, textvariable=self.batch_dir, font=('Consolas', 9)).pack(fill='x', pady=5)
        
        btn_frame = ttk.Frame(dir_frame)
        btn_frame.pack(fill='x')
        
        ttk.Button(btn_frame, text="Browse...", command=self._browse_batch_dir).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Run Batch Analysis", command=self._run_batch_analysis).pack(side='left', padx=5)
        
        # Progress
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_label = tk.StringVar(value="Ready to analyze...")
        
        ttk.Label(self.batch_tab, textvariable=self.progress_label, font=('Consolas', 9)).pack(pady=5)
        self.progress_bar = ttk.Progressbar(self.batch_tab, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', pady=5)
        
        # Batch results
        batch_results_frame = ttk.LabelFrame(self.batch_tab, text="Batch Analysis Log", padding=10)
        batch_results_frame.pack(fill='both', expand=True, pady=10)
        
        self.batch_text = tk.Text(batch_results_frame, font=('Consolas', 10), 
                                  wrap='word', state='disabled', height=20)
        self.batch_text.pack(side='left', fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(batch_results_frame, orient='vertical', 
                                  command=self.batch_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.batch_text.config(yscrollcommand=scrollbar.set)
        
        self.batch_text.tag_configure('header', font=('Consolas', 11, 'bold'))
        self.batch_text.tag_configure('success', foreground='#006600')
        self.batch_text.tag_configure('info', foreground='#0066CC')
    
    def _browse_file(self):
        initial_dir = os.path.join(os.path.dirname(__file__), 'results')
        if not os.path.exists(initial_dir):
            initial_dir = os.path.dirname(__file__)
        
        filepath = filedialog.askopenfilename(
            title="Select Results JSON File",
            initialdir=initial_dir,
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            self.current_file.set(filepath)
            self.file_label.config(foreground='black')
    
    def _browse_batch_dir(self):
        initial_dir = os.path.join(os.path.dirname(__file__), 'results')
        
        dirpath = filedialog.askdirectory(
            title="Select Results Directory",
            initialdir=initial_dir
        )
        
        if dirpath:
            self.batch_dir.set(dirpath)
    
    def _analyze(self):
        filepath = self.current_file.get()
        
        if filepath == "No file selected" or not os.path.exists(filepath):
            messagebox.showerror("Error", "Please select a valid results file first.")
            return
        
        try:
            self.analyzer = ResultsAnalyzer(filepath)
            self._display_results()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze file:\n{str(e)}")
    
    def _run_batch_analysis(self):
        base_dir = self.batch_dir.get()
        
        if not os.path.exists(base_dir):
            messagebox.showerror("Error", "Results directory does not exist.")
            return
        
        # Clear log
        self.batch_text.config(state='normal')
        self.batch_text.delete('1.0', 'end')
        
        self._batch_append("=" * 60 + "\n", 'header')
        self._batch_append("BATCH ANALYSIS STARTED\n", 'header')
        self._batch_append("=" * 60 + "\n\n")
        self._batch_append(f"Base directory: {base_dir}\n")
        self._batch_append(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        self.batch_text.config(state='disabled')
        self.root.update()
        
        def progress_callback(current, total, message):
            pct = (current / total * 100) if total > 0 else 0
            self.progress_var.set(pct)
            self.progress_label.set(f"{current}/{total} - {message}")
            
            self.batch_text.config(state='normal')
            self._batch_append(f"  ✓ {message}\n", 'success')
            self.batch_text.config(state='disabled')
            self.root.update()
        
        try:
            batch_analyzer = BatchAnalyzer(base_dir)
            analyzed_files = batch_analyzer.run_batch_analysis(progress_callback)
            
            self.batch_text.config(state='normal')
            self._batch_append("\n" + "=" * 60 + "\n", 'header')
            self._batch_append("BATCH ANALYSIS COMPLETE\n", 'header')
            self._batch_append("=" * 60 + "\n\n")
            
            self._batch_append(f"Results files analyzed: {len(analyzed_files['results'])}\n", 'info')
            self._batch_append(f"Summary files analyzed: {len(analyzed_files['summaries'])}\n", 'info')
            self._batch_append(f"\nOutput directory: {batch_analyzer.output_dir}\n", 'info')
            
            self.batch_text.config(state='disabled')
            
            self.progress_label.set("Batch analysis complete!")
            messagebox.showinfo("Success", 
                              f"Batch analysis complete!\n\n"
                              f"Results analyzed: {len(analyzed_files['results'])}\n"
                              f"Summaries analyzed: {len(analyzed_files['summaries'])}\n\n"
                              f"Output saved to:\n{batch_analyzer.output_dir}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Batch analysis failed:\n{str(e)}")
            self.progress_label.set("Error during analysis")
    
    def _batch_append(self, text: str, tag: str = None):
        if tag:
            self.batch_text.insert('end', text, tag)
        else:
            self.batch_text.insert('end', text)
        self.batch_text.see('end')
    
    def _display_results(self):
        if not self.analyzer:
            return
        
        # Enable editing
        self.results_text.config(state='normal')
        self.results_text.delete('1.0', 'end')
        
        # Get stats
        basic = self.analyzer.get_basic_stats()
        gold = self.analyzer.analyze_gold_docs()
        dist = self.analyzer.get_gold_distribution()
        
        # Display basic stats
        self._append("=" * 60 + "\n", 'header')
        self._append("BASIC STATISTICS\n", 'header')
        self._append("=" * 60 + "\n\n")
        
        self._append(f"File: {os.path.basename(self.analyzer.results_path)}\n")
        self._append(f"Experiment Type: {basic['experiment_type'].upper()}\n", 'metric')
        self._append(f"k value: {basic['k']}\n")
        
        if 'total_docs' in basic:
            self._append(f"Total docs per query: {basic['total_docs']}\n")
            self._append(f"Noise/Hard-neg docs: {basic['num_noise']}\n")
        
        self._append(f"Total questions: {basic['total_questions']}\n")
        self._append(f"Correct answers (matches): {basic['matches']}\n")
        self._append(f"Exact Match score: {basic['exact_match']:.4f} ({basic['exact_match']*100:.2f}%)\n", 'metric')
        self._append(f"Errors: {basic['errors']}\n")
        
        if 'gold_recall_rate' in basic:
            self._append(f"Gold Recall Rate: {basic['gold_recall_rate']:.4f} ({basic['gold_recall_rate']*100:.2f}%)\n", 'highlight')
            self._append(f"Gold EM When Retrieved: {basic['gold_em_when_retrieved']:.4f} ({basic['gold_em_when_retrieved']*100:.2f}%)\n", 'highlight')
        
        if 'avg_oracle_contexts' in basic:
            self._append(f"Avg Oracle Contexts: {basic['avg_oracle_contexts']:.3f}\n")
        
        self._append(f"Timestamp: {basic['timestamp']}\n\n")
        
        # Skip gold doc analysis for exp2 (oracle)
        if basic['experiment_type'] != 'exp2':
            # Display gold doc analysis
            self._append("=" * 60 + "\n", 'header')
            self._append("GOLDEN DOCUMENT ANALYSIS\n", 'header')
            self._append("=" * 60 + "\n\n")
            
            self._append("For CORRECT answers (matches):\n", 'header')
            self._append("-" * 40 + "\n")
            self._append(f"Total matches: {gold['total_matches']}\n")
            self._append(f"Matches with >= 1 gold doc in retrieved: {gold['matches_with_at_least_1_gold']}\n", 'highlight')
            self._append(f"Matches without any gold doc: {gold['matches_without_any_gold']}\n", 'warning')
            self._append(f"Matches with ALL gold docs in retrieved: {gold['matches_with_all_gold']}\n")
            self._append("\n")
            
            frac = gold['fraction_matches_with_gold']
            self._append(f">>> Fraction of matches with at least 1 gold doc: ", 'metric')
            self._append(f"{frac:.4f} ({frac*100:.2f}%)\n\n", 'highlight')
            
            frac_all = gold['fraction_matches_with_all_gold']
            self._append(f">>> Fraction of matches with ALL gold docs: ", 'metric')
            self._append(f"{frac_all:.4f} ({frac_all*100:.2f}%)\n\n", 'highlight')
            
            self._append("For INCORRECT answers (non-matches):\n", 'header')
            self._append("-" * 40 + "\n")
            self._append(f"Total non-matches: {gold['total_non_matches']}\n")
            self._append(f"Non-matches with >= 1 gold doc: {gold['non_matches_with_at_least_1_gold']}\n")
            self._append(f"Non-matches with ALL gold docs: {gold['non_matches_with_all_gold']}\n")
            
            frac_nm = gold['fraction_non_matches_with_gold']
            self._append(f"\n>>> Fraction of non-matches with at least 1 gold doc: ", 'metric')
            self._append(f"{frac_nm:.4f} ({frac_nm*100:.2f}%)\n\n")
            
            # Distribution
            self._append("=" * 60 + "\n", 'header')
            self._append("GOLD DOCS IN TOP-K DISTRIBUTION\n", 'header')
            self._append("=" * 60 + "\n\n")
            
            for num_gold, count in dist.items():
                pct = count / len(self.analyzer.results) * 100
                bar = "#" * int(pct / 2)
                self._append(f"{num_gold} gold doc(s): {count:4d} ({pct:5.1f}%) {bar}\n")
        else:
            self._append("=" * 60 + "\n", 'header')
            self._append("NOTE: Oracle experiment (Exp2) - no retrieval analysis\n", 'warning')
            self._append("=" * 60 + "\n")
        
        # Disable editing
        self.results_text.config(state='disabled')
    
    def _append(self, text: str, tag: str = None):
        if tag:
            self.results_text.insert('end', text, tag)
        else:
            self.results_text.insert('end', text)
    
    def run(self):
        self.root.mainloop()


def main():
    ui = ResultsAnalyzerUI()
    ui.run()


if __name__ == "__main__":
    main()