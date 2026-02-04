"""
PDF generation service for meeting summaries.
Uses reportlab for PDF creation.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    ListFlowable,
    ListItem,
)

from ..models import MeetingArtifacts


class PDFGeneratorService:
    """Generates PDF summaries from meeting artifacts."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Configure custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=self.styles['Heading2'],
            textColor=colors.HexColor('#2563eb'),
            spaceAfter=12,
        ))
        self.styles.add(ParagraphStyle(
            name='ItemTitle',
            parent=self.styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=11,
            spaceAfter=4,
        ))
        self.styles.add(ParagraphStyle(
            name='ItemDetail',
            parent=self.styles['Normal'],
            fontSize=10,
            leftIndent=20,
            textColor=colors.HexColor('#4b5563'),
        ))
    
    async def generate(self, artifacts: MeetingArtifacts, filename: str | None = None) -> Path:
        """
        Generate a PDF summary from meeting artifacts.
        
        Args:
            artifacts: The extracted meeting artifacts
            filename: Optional filename (generated if not provided)
            
        Returns:
            Path to the generated PDF file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"meeting_summary_{timestamp}.pdf"
        
        output_path = self.output_dir / filename
        
        # Run PDF generation in thread pool to not block async
        await asyncio.to_thread(self._create_pdf, artifacts, output_path)
        
        return output_path
    
    def _create_pdf(self, artifacts: MeetingArtifacts, output_path: Path):
        """Create the actual PDF document."""
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )
        
        story = []
        
        # Title
        story.append(Paragraph(
            f"📋 {artifacts.meeting_title}",
            self.styles['Title']
        ))
        story.append(Spacer(1, 12))
        
        # Meeting info
        info_data = [
            ["Date:", artifacts.meeting_date.strftime("%B %d, %Y at %H:%M")],
            ["Participants:", ", ".join(artifacts.participants) or "Not specified"],
        ]
        if artifacts.duration_minutes:
            info_data.append(["Duration:", f"{artifacts.duration_minutes} minutes"])
        
        info_table = Table(info_data, colWidths=[1.5*inch, 4.5*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#6b7280')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 20))
        
        # Summary
        if artifacts.summary:
            story.append(Paragraph("Summary", self.styles['SectionTitle']))
            story.append(Paragraph(artifacts.summary, self.styles['Normal']))
            story.append(Spacer(1, 20))
        
        # User Stories
        if artifacts.user_stories:
            story.append(Paragraph("📖 User Stories", self.styles['SectionTitle']))
            for i, us in enumerate(artifacts.user_stories, 1):
                story.append(Paragraph(
                    f"{i}. {us.title} ({us.story_points or '?'} pts, {us.priority.value})",
                    self.styles['ItemTitle']
                ))
                story.append(Paragraph(
                    f"As a <b>{us.as_a}</b>, I want <b>{us.i_want}</b>, so that <b>{us.so_that}</b>",
                    self.styles['ItemDetail']
                ))
                if us.acceptance_criteria:
                    criteria_items = [
                        ListItem(Paragraph(c, self.styles['Normal']), leftIndent=35)
                        for c in us.acceptance_criteria
                    ]
                    story.append(ListFlowable(criteria_items, bulletType='bullet', start='•'))
                story.append(Spacer(1, 8))
            story.append(Spacer(1, 12))
        
        # Tasks
        if artifacts.tasks:
            story.append(Paragraph("✅ Tasks", self.styles['SectionTitle']))
            task_data = [["Task", "Assignee", "Priority", "Due"]]
            for task in artifacts.tasks:
                task_data.append([
                    task.title[:40] + "..." if len(task.title) > 40 else task.title,
                    task.assignee or "-",
                    task.priority.value.title(),
                    task.due_date or "-",
                ])
            
            task_table = Table(task_data, colWidths=[3*inch, 1.2*inch, 0.9*inch, 1*inch])
            task_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563eb')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e5e7eb')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(task_table)
            story.append(Spacer(1, 20))
        
        # Decisions
        if artifacts.decisions:
            story.append(Paragraph("🎯 Decisions", self.styles['SectionTitle']))
            for d in artifacts.decisions:
                story.append(Paragraph(f"• {d.title}", self.styles['ItemTitle']))
                story.append(Paragraph(d.description, self.styles['ItemDetail']))
                if d.rationale:
                    story.append(Paragraph(
                        f"<i>Rationale: {d.rationale}</i>",
                        self.styles['ItemDetail']
                    ))
                story.append(Spacer(1, 8))
            story.append(Spacer(1, 12))
        
        # Blockers
        if artifacts.blockers:
            story.append(Paragraph("🚧 Blockers", self.styles['SectionTitle']))
            for b in artifacts.blockers:
                story.append(Paragraph(f"• {b.title}", self.styles['ItemTitle']))
                story.append(Paragraph(b.description, self.styles['ItemDetail']))
                if b.owner:
                    story.append(Paragraph(f"Owner: {b.owner}", self.styles['ItemDetail']))
                if b.resolution_plan:
                    story.append(Paragraph(
                        f"Resolution: {b.resolution_plan}",
                        self.styles['ItemDetail']
                    ))
                story.append(Spacer(1, 8))
            story.append(Spacer(1, 12))
        
        # Action Items
        if artifacts.action_items:
            story.append(Paragraph("📌 Action Items", self.styles['SectionTitle']))
            for a in artifacts.action_items:
                assignee = f" ({a.assignee})" if a.assignee else ""
                due = f" - Due: {a.due_date}" if a.due_date else ""
                story.append(Paragraph(
                    f"• {a.description}{assignee}{due}",
                    self.styles['Normal']
                ))
            story.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(story)
