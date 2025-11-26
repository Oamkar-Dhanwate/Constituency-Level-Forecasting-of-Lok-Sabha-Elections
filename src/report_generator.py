# src/report_generator.py
import pandas as pd
import plotly.io as pio
from fpdf import FPDF
from datetime import datetime
import io
import warnings

# Suppress FPDF font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='fpdf')

class PDF(FPDF):
    """Custom PDF class to handle header and footer"""
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Election Analysis Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        page_num = self.page_no()
        self.cell(0, 10, f'Page {page_num}', 0, 0, 'C')

def generate_pdf_report(selections, metrics, plots, prediction, ai_conclusion):
    """
    Generates a PDF report using FPDF.
    'plots' is expected to be a dictionary containing plot bytes, e.g., plots['map_chart']
    """
    
    pdf = PDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font('Arial', '', 12)

    # 1. Title
    pdf.set_font('Arial', 'B', 18)
    pdf.cell(0, 10, 'Election Analysis Report', 0, 1, 'C')
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 8, f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, 'C')
    pdf.ln(10)

    # 2. Selections
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Selections', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f"  Year: {selections.get('year', 'All')}", 0, 1)
    pdf.cell(0, 8, f"  State: {selections.get('state', 'All States')}", 0, 1)
    pdf.ln(5)

    # 3. Key Metrics
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Key Metrics', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f"  Total Votes: {metrics.get('total_votes', 'N/A')}", 0, 1)
    pdf.cell(0, 8, f"  Total Constituencies: {metrics.get('total_constituencies', 'N/A')}", 0, 1)
    pdf.ln(5)

    # 4. AI-Powered Prediction
    if prediction.get('prediction_made', False):
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'AI-Powered Prediction', 0, 1)
        pdf.set_font('Arial', '', 11)
        # Use multi_cell for inputs, as it can wrap text
        pdf.multi_cell(0, 8, f"  Inputs: {str(prediction.get('inputs', 'N/A'))}", 0, 1)
        pdf.cell(0, 8, f"  Predicted Vote Share: {prediction.get('result', 0.0):.2f}%", 0, 1)
        pdf.ln(5)

    # 5. AI Conclusion
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'AI Conclusion', 0, 1)
    pdf.set_font('Arial', '', 11)
    # multi_cell automatically handles newlines
    pdf.multi_cell(0, 8, ai_conclusion)
    pdf.ln(10)

    # 6. Plots
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Visualizations', 0, 1)
    pdf.set_font('Arial', '', 11)
    
    # Iterate through the plots dict (from your app_02.py)
    # This expects the plots to be PNG bytes
    plot_width = 190 # Width in mm for an A4 page with margins

    if 'map_chart' in plots and plots['map_chart']:
        pdf.add_page() # Add a new page for the first big plot
        pdf.image(io.BytesIO(plots['map_chart']), x=10, w=plot_width, type='png')
        pdf.cell(0, 5, 'Geographical Map', 0, 1, 'C')
        pdf.ln(5)
        
    if 'trend_chart' in plots and plots['trend_chart']:
        pdf.add_page()
        pdf.image(io.BytesIO(plots['trend_chart']), x=10, w=plot_width, type='png')
        pdf.cell(0, 5, 'Historical Trend', 0, 1, 'C')
        pdf.ln(5)
    
    if 'incumbency_chart' in plots and plots['incumbency_chart']:
        pdf.add_page()
        pdf.image(io.BytesIO(plots['incumbency_chart']), x=10, w=plot_width, type='png')
        pdf.cell(0, 5, 'Incumbency Analysis', 0, 1, 'C')
        pdf.ln(5)
    
    # 7. Generate PDF bytes in memory
    # Use latin-1 encoding for FPDF's byte output
    return pdf.output(dest='S').encode('latin1')