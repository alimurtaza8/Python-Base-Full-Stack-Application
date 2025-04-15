import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, roc_auc_score,
    classification_report, precision_score, recall_score
)
import time
import os
from datetime import datetime

class ModelEvaluator:
    """Class for evaluating and comparing EPC prediction models."""
    
    def __init__(self, output_dir='evaluation_results'):
        """
        Initialize the model evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        self.results = {}
        self.execution_times = {}
        self.models = {}
        self.class_mapping = None
        
        os.makedirs(output_dir, exist_ok=True)
    
    def set_class_mapping(self, class_mapping):
        """
        Set the mapping between numeric classes and EPC ratings.
        
        Args:
            class_mapping: Dictionary mapping numeric class to EPC rating
        """
        self.class_mapping = class_mapping
    
    def register_model(self, model_name, model_object):
        """
        Register a model for evaluation.
        
        Args:
            model_name: Name of the model
            model_object: Model object with predict and predict_proba methods
        """
        self.models[model_name] = model_object
    
    def evaluate_model(self, model_name, X_test, y_test, model=None):
        """
        Evaluate a single model and store the results.
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: True labels
            model: Model object (optional, if not registered)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if model is None:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not registered or provided.")
            model = self.models[model_name]
        
        # Measure execution time
        start_time = time.time()
        y_pred = model.predict(X_test)
        execution_time = time.time() - start_time
        
        # If model supports probability prediction
        try:
            y_prob = model.predict_proba(X_test)
            has_proba = True
        except (AttributeError, NotImplementedError):
            y_prob = None
            has_proba = False
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'execution_time': execution_time
        }
        
        # Calculate ROC-AUC if probability predictions are available
        if has_proba and y_prob is not None:
            # For multi-class, use one-vs-rest approach
            if y_prob.shape[1] > 2:
                metrics['roc_auc'] = roc_auc_score(
                    pd.get_dummies(y_test), y_prob, 
                    multi_class='ovr', average='macro'
                )
            else:
                metrics['roc_auc'] = roc_auc_score(y_test, y_prob[:, 1])
        
        # Store results
        self.results[model_name] = metrics
        self.execution_times[model_name] = execution_time
        
        return metrics
    
    def evaluate_all_models(self, X_test, y_test):
        """
        Evaluate all registered models.
        
        Args:
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary of evaluation results for all models
        """
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            self.evaluate_model(model_name, X_test, y_test, model)
        
        return self.results
    
    def compare_models(self, metric='f1_weighted'):
        """
        Compare models based on a specific metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            raise ValueError("No evaluation results available. Call evaluate_model() first.")
        
        comparison = {}
        for model_name, metrics in self.results.items():
            comparison[model_name] = {
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'f1_weighted': metrics['f1_weighted'],
                'precision_macro': metrics['precision_macro'],
                'recall_macro': metrics['recall_macro'],
                'execution_time': metrics['execution_time']
            }
            
            if 'roc_auc' in metrics:
                comparison[model_name]['roc_auc'] = metrics['roc_auc']
        
        df_comparison = pd.DataFrame(comparison).T
        
        # Sort by the specified metric
        df_comparison = df_comparison.sort_values(by=metric, ascending=False)
        
        return df_comparison
    
    def plot_confusion_matrices(self, figsize=(15, 10)):
        """
        Plot confusion matrices for all evaluated models.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.results:
            raise ValueError("No evaluation results available. Call evaluate_model() first.")
        
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        # Handle single model case
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, metrics) in enumerate(self.results.items()):
            cm = metrics['confusion_matrix']
            
            # Use class mapping if available
            if self.class_mapping:
                labels = [self.class_mapping.get(i, str(i)) for i in range(cm.shape[0])]
            else:
                labels = range(cm.shape[0])
            
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels,
                ax=axes[i]
            )
            axes[i].set_title(f"Confusion Matrix - {model_name}")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("True")
        
        plt.tight_layout()
        
        # Save the figure
        save_path = os.path.join(self.output_dir, 'confusion_matrices.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved at {save_path}")
        
        return fig
    
    def plot_metrics_comparison(self, metrics=None, figsize=(12, 8)):
        """
        Plot comparison of multiple metrics across models.
        
        Args:
            metrics: List of metrics to compare
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.results:
            raise ValueError("No evaluation results available. Call evaluate_model() first.")
        
        if metrics is None:
            metrics = ['accuracy', 'f1_weighted', 'precision_macro', 'recall_macro']
            if 'roc_auc' in next(iter(self.results.values())):
                metrics.append('roc_auc')
        
        # Extract metrics for comparison
        comparison_data = {}
        for model_name, results in self.results.items():
            comparison_data[model_name] = {
                metric: results[metric] for metric in metrics 
                if metric in results and metric != 'confusion_matrix'
            }
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        df_comparison.T.plot(kind='bar', ax=ax)
        
        ax.set_title("Model Comparison by Metrics")
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.legend(title="Metric")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the figure
        save_path = os.path.join(self.output_dir, 'metrics_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved at {save_path}")
        
        return fig
    
    def plot_execution_times(self, figsize=(10, 6)):
        """
        Plot model execution times.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.execution_times:
            raise ValueError("No execution times available. Call evaluate_model() first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        models = list(self.execution_times.keys())
        times = list(self.execution_times.values())
        
        # Sort by execution time
        sorted_indices = np.argsort(times)
        sorted_models = [models[i] for i in sorted_indices]
        sorted_times = [times[i] for i in sorted_indices]
        
        bars = ax.barh(sorted_models, sorted_times, color='skyblue')
        
        # Add execution time values at the end of each bar
        for i, (model, time_val) in enumerate(zip(sorted_models, sorted_times)):
            ax.text(time_val + 0.001, i, f"{time_val:.4f}s", va='center')
        
        ax.set_title("Model Inference Time Comparison")
        ax.set_xlabel("Execution Time (seconds)")
        ax.set_ylabel("Model")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Save the figure
        save_path = os.path.join(self.output_dir, 'execution_times.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Execution times plot saved at {save_path}")
        
        return fig
    
    def generate_report(self):
        """
        Generate a comprehensive evaluation report.
        
        Returns:
            DataFrame containing the report
        """
        if not self.results:
            raise ValueError("No evaluation results available. Call evaluate_model() first.")
        
        # Create comparison DataFrame
        comparison_df = self.compare_models()
        
        # Generate full report with summary
        summary = {
            'best_accuracy_model': comparison_df['accuracy'].idxmax(),
            'best_accuracy_value': comparison_df['accuracy'].max(),
            'best_f1_model': comparison_df['f1_weighted'].idxmax(),
            'best_f1_value': comparison_df['f1_weighted'].max(),
            'fastest_model': comparison_df['execution_time'].idxmin(),
            'fastest_time': comparison_df['execution_time'].min()
        }
        
        if 'roc_auc' in comparison_df.columns:
            summary['best_roc_auc_model'] = comparison_df['roc_auc'].idxmax()
            summary['best_roc_auc_value'] = comparison_df['roc_auc'].max()
        
        # Determine overall best model
        # Weighted average of normalized scores
        weights = {
            'accuracy': 0.25,
            'f1_weighted': 0.35,
            'precision_macro': 0.15,
            'recall_macro': 0.15,
            'execution_time': 0.1
        }
        
        if 'roc_auc' in comparison_df.columns:
            weights = {k: v * 0.9 for k, v in weights.items()}  # Scale down to allow for roc_auc
            weights['roc_auc'] = 0.1
        
        # Normalize metrics (higher is better, except execution_time)
        normalized_df = comparison_df.copy()
        for column in normalized_df.columns:
            if column == 'execution_time':
                # Lower is better for execution time
                if normalized_df[column].max() != normalized_df[column].min():
                    normalized_df[column] = 1 - ((normalized_df[column] - normalized_df[column].min()) / 
                                            (normalized_df[column].max() - normalized_df[column].min()))
                else:
                    normalized_df[column] = 1  # All equal
            else:
                # Higher is better for other metrics
                if normalized_df[column].max() != normalized_df[column].min():
                    normalized_df[column] = (normalized_df[column] - normalized_df[column].min()) / \
                                        (normalized_df[column].max() - normalized_df[column].min())
                else:
                    normalized_df[column] = 1  # All equal
        
        # Calculate weighted score
        for model in normalized_df.index:
            score = 0
            for metric, weight in weights.items():
                if metric in normalized_df.columns:
                    score += normalized_df.loc[model, metric] * weight
            normalized_df.loc[model, 'overall_score'] = score
        
        best_model = normalized_df['overall_score'].idxmax()
        summary['best_overall_model'] = best_model
        summary['best_overall_score'] = normalized_df.loc[best_model, 'overall_score']
        
        # Create a report DataFrame
        report_df = comparison_df.copy()
        report_df['overall_score'] = normalized_df['overall_score']
        report_df = report_df.sort_values(by='overall_score', ascending=False)
        
        # Save the report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f'evaluation_report_{timestamp}.csv')
        report_df.to_csv(report_path)
        print(f"Evaluation report saved at {report_path}")
        
        # Generate summary as a DataFrame
        summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
        summary_path = os.path.join(self.output_dir, f'summary_{timestamp}.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary report saved at {summary_path}")
        
        return {
            'comparison': report_df,
            'summary': summary_df
        }