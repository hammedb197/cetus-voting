"""Analytics module for processing and visualizing voting data."""

import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, UTC
from .types import VotingPowerAnalysis, VotingSummary
from .constants import VOTE_TYPES, VOTE_COLORS, PIE_CHART_COLORS

class VotingAnalytics:
    """Class for analyzing and visualizing voting data."""
    
    @staticmethod
    def prepare_time_based_data(
        df: pd.DataFrame,
        view_type: str = "Hourly"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare time-based voting data for visualization.
        
        Args:
            df: DataFrame containing voting data
            view_type: Type of time aggregation ("Hourly" or "Daily")
            
        Returns:
            Tuple containing cumulative votes DataFrame and formatted dates
        """
        df_time = df.copy()
        df_time['date'] = pd.to_datetime(df_time['timestamp'])
        df_time = df_time.sort_values('date')
        
        # Create time-based aggregation
        freq = 'h' if view_type == "Hourly" else 'D'
        
        # Group by time and vote, then count
        vote_counts = df_time.groupby([
            pd.Grouper(key='date', freq=freq),
            'vote'
        ]).size().reset_index(name='count')
        
        # Pivot the data
        vote_pivot = vote_counts.pivot(
            index='date',
            columns='vote',
            values='count'
        ).fillna(0)
        
        # Calculate cumulative sums
        cumulative_votes = vote_pivot.cumsum()
        
        # Ensure all vote types exist
        for vote_type in VOTE_TYPES:
            if vote_type not in cumulative_votes.columns:
                cumulative_votes[vote_type] = 0
        
        # Format dates for display
        date_format = "%m-%d %H:%M" if view_type == "Hourly" else "%m-%d"
        formatted_dates = cumulative_votes.index.strftime(date_format)
        
        return cumulative_votes, formatted_dates

    @staticmethod
    def create_timeline_figure(
        cumulative_votes: pd.DataFrame,
        formatted_dates: List[str]
    ) -> go.Figure:
        """Create timeline visualization for voting data.
        
        Args:
            cumulative_votes: DataFrame with cumulative vote counts
            formatted_dates: List of formatted date strings
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        for vote_type in VOTE_TYPES:
            fig.add_trace(go.Scatter(
                name=vote_type,
                x=formatted_dates,
                y=cumulative_votes[vote_type],
                mode='lines',
                line=dict(
                    color=VOTE_COLORS[vote_type],
                    width=3,
                    shape='spline',
                    smoothing=0.3
                ),
                hovertemplate=(
                    f"<b>{vote_type}</b><br>" +
                    "Time: %{x}<br>" +
                    "Votes: %{y}<br>" +
                    "<extra></extra>"
                )
            ))
        
        fig.update_layout(
            title='Voting Breakdown',
            xaxis_title=None,
            yaxis_title="Number of Votes",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                showline=False,
                linecolor='rgba(128, 128, 128, 0.2)',
                linewidth=1,
                tickfont=dict(size=12),
                tickangle=-45,
                type='category',
                categoryorder='array',
                categoryarray=formatted_dates
            ),
            yaxis=dict(
                gridwidth=1,
                showline=True,
                tickfont=dict(size=12),
                title_text="Number of Votes",
                rangemode='nonnegative'
            ),
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig

    @staticmethod
    def create_power_distribution_figure(
        power_data: Dict[str, float],
        remaining_power: float
    ) -> Tuple[go.Figure, float, float]:
        """Create pie chart for voting power distribution.
        
        Args:
            power_data: Dictionary containing voting power by vote type
            remaining_power: Remaining voting power
            
        Returns:
            Tuple of (Plotly figure, yes percentage, no percentage)
        """
        fig = go.Figure()
        
        power_labels = ['Yes', 'No', 'Abstain', 'Not Voted']
        power_values = [
            power_data['Yes'],
            power_data['No'],
            power_data['Abstain'],
            remaining_power
        ]
        
        # Calculate percentages
        total_power = sum(power_values)
        yes_percent = (power_data['Yes'] / total_power * 100) if total_power > 0 else 0
        no_percent = (power_data['No'] / total_power * 100) if total_power > 0 else 0
        
        fig.add_trace(go.Pie(
            labels=power_labels,
            values=power_values,
            hole=.4,
            textinfo='label+percent',
            marker=dict(colors=PIE_CHART_COLORS),
            domain=dict(x=[0, 1], y=[0, 1]),
            showlegend=True
        ))
        
        fig.update_layout(
            title=f"Current Distribution (Yes: {yes_percent:.1f}% | No: {no_percent:.1f}%)",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=400
        )
        
        return fig, yes_percent, no_percent

    @staticmethod
    def calculate_vote_metrics(
        power_data: Dict[str, float],
        remaining_power: float
    ) -> Dict[str, Tuple[float, str]]:
        """Calculate voting metrics and impact analysis.
        
        Args:
            power_data: Dictionary containing voting power by vote type
            remaining_power: Remaining voting power
            
        Returns:
            Dictionary of metrics with their values and help text
        """
        return {
            "Yes Power": (power_data['Yes'], "Total voting power supporting the proposal"),
            "No Power": (power_data['No'], "Total voting power against the proposal"),
            "Remaining": (remaining_power, "Voting power yet to be cast"),
            "Difference": (abs(power_data['Yes'] - power_data['No']), 
                         "Absolute difference between Yes and No votes")
        }

    @staticmethod
    def prepare_validator_activity(
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Prepare validator voting activity data.
        
        Args:
            df: DataFrame containing voting data
            
        Returns:
            Formatted DataFrame with validator activity
        """
        if df.empty:
            return pd.DataFrame()
            
        vote_df = df[['timestamp', 'validator_name', 'validator_address', 'vote']].copy()
        vote_df['timestamp'] = pd.to_datetime(vote_df['timestamp'])
        vote_df = vote_df.sort_values('timestamp', ascending=False)
        
        # Rename columns
        vote_df.columns = ['Time', 'Validator', 'Address', 'Vote']
        vote_df['Time'] = vote_df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return vote_df

    @staticmethod
    def prepare_non_voted_validators(
        not_voted_validators: List[Dict[str, Any]],
        total_voting_power: float
    ) -> Optional[pd.DataFrame]:
        """Prepare data for non-voted validators.
        
        Args:
            not_voted_validators: List of validators who haven't voted
            total_voting_power: Total voting power in the system
            
        Returns:
            Formatted DataFrame with non-voted validator information
        """
        if not not_voted_validators:
            return None
            
        not_voted_df = pd.DataFrame(not_voted_validators)
        
        # Ensure voting_power is numeric
        not_voted_df['voting_power'] = pd.to_numeric(not_voted_df['voting_power'])
        
        # Calculate power percentage
        not_voted_df['power_percentage'] = (
            not_voted_df['voting_power'] / total_voting_power * 100
        ).round(2)
        
        # Format the DataFrame
        display_df = not_voted_df.copy()
        display_df = display_df.rename(columns={
            'name': 'Validator',
            'address': 'Address',
            'voting_power': 'Voting Power',
            'power_percentage': 'Power %'
        })
        
        # Format the Power % column
        display_df['Power %'] = display_df['Power %'].astype(str) + '%'
        
        # Sort by voting power descending
        return display_df.sort_values('Voting Power', ascending=False)

    @staticmethod
    def format_power(value: float) -> str:
        """Format large power numbers for display.
        
        Args:
            value: Number to format
            
        Returns:
            Formatted string representation
        """
        if value >= 1_000_000:
            return f"{value/1_000_000:.1f}M"
        elif value >= 1_000:
            return f"{value/1_000:.1f}K"
        return str(value) 