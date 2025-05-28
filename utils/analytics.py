"""Analytics module for processing governance data."""

from typing import Dict, List, Any, Tuple
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

class GovernanceAnalytics:
    """Analytics for governance data."""
    
    @staticmethod
    def calculate_vote_distribution(df: pd.DataFrame) -> Dict[str, int]:
        """Calculate the distribution of votes.
        
        Args:
            df: DataFrame containing vote data
            
        Returns:
            Dictionary with vote counts by type
        """
        if df.empty:
            return {'Yes': 0, 'No': 0, 'Abstain': 0}
            
        return df['vote'].value_counts().to_dict()
    
    @staticmethod
    def calculate_voter_participation(df: pd.DataFrame) -> float:
        """Calculate voter participation rate.
        
        Args:
            df: DataFrame containing vote data
            
        Returns:
            Participation rate as a percentage
        """
        if df.empty:
            return 0.0
            
        unique_voters = df['validator_address'].nunique()
        # TODO: Get total possible validators from the network
        total_validators = 100  # Placeholder
        
        return (unique_voters / total_validators) * 100
    
    @staticmethod
    def create_vote_timeline(df: pd.DataFrame) -> go.Figure:
        """Create a timeline visualization of votes.
        
        Args:
            df: DataFrame containing vote data
            
        Returns:
            Plotly figure object
        """
        if df.empty:
            return go.Figure()
            
        # Group by timestamp and vote type
        timeline_data = df.groupby(
            [pd.Grouper(key='timestamp', freq='1H'), 'vote']
        ).size().unstack(fill_value=0).cumsum()
        
        fig = go.Figure()
        
        colors = {
            'Yes': '#00CC96',
            'No': '#EF553B',
            'Abstain': '#636EFA'
        }
        
        for vote_type in timeline_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=timeline_data.index,
                    y=timeline_data[vote_type],
                    name=vote_type,
                    mode='lines',
                    line=dict(
                        color=colors.get(vote_type, '#000000'),
                        width=2
                    ),
                    fill='tonexty'
                )
            )
            
        fig.update_layout(
            title='Cumulative Votes Over Time',
            xaxis_title='Time',
            yaxis_title='Number of Votes',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    @staticmethod
    def create_vote_distribution_pie(df: pd.DataFrame) -> go.Figure:
        """Create a pie chart of vote distribution.
        
        Args:
            df: DataFrame containing vote data
            
        Returns:
            Plotly figure object
        """
        if df.empty:
            return go.Figure()
            
        vote_counts = df['vote'].value_counts()
        colors = {
            'Yes': '#00CC96',
            'No': '#EF553B',
            'Abstain': '#636EFA'
        }
        
        fig = go.Figure(data=[
            go.Pie(
                labels=vote_counts.index,
                values=vote_counts.values,
                hole=.3,
                marker=dict(colors=[colors.get(v, '#000000') for v in vote_counts.index])
            )
        ])
        
        fig.update_layout(
            title='Vote Distribution',
            showlegend=True,
            legend=dict(orientation="h"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    @staticmethod
    def get_top_voters(df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
        """Get the most active voters.
        
        Args:
            df: DataFrame containing vote data
            limit: Number of top voters to return
            
        Returns:
            DataFrame with top voters and their vote counts
        """
        if df.empty:
            return pd.DataFrame()
            
        # Get vote counts by validator
        voter_stats = (
            df.groupby('validator_address')
            .agg({
                'vote': ['count', lambda x: x.value_counts().to_dict()]
            })
            .reset_index()
        )
        
        # Flatten the multi-level columns
        voter_stats.columns = ['validator_address', 'vote_count', 'vote_breakdown']
        
        # Sort by vote count
        voter_stats = voter_stats.sort_values('vote_count', ascending=False).head(limit)
        
        # Calculate percentage
        total_votes = voter_stats['vote_count'].sum()
        voter_stats['percentage'] = (voter_stats['vote_count'] / total_votes * 100).round(1)
        
        return voter_stats
    
    @staticmethod
    def get_vote_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Get a summary of voting activity.
        
        Args:
            df: DataFrame containing vote data
            
        Returns:
            Dictionary containing voting summary statistics
        """
        if df.empty:
            return {
                'total_votes': 0,
                'unique_voters': 0,
                'vote_distribution': {'Yes': 0, 'No': 0, 'Abstain': 0},
                'participation_rate': 0.0,
                'voting_power': {
                    'total': 0,
                    'voted': 0,
                    'percentage': 0.0
                }
            }
            
        vote_dist = GovernanceAnalytics.calculate_vote_distribution(df)
        participation = GovernanceAnalytics.calculate_voter_participation(df)
        
        return {
            'total_votes': len(df),
            'unique_voters': df['validator_address'].nunique(),
            'vote_distribution': vote_dist,
            'participation_rate': participation,
            'voting_power': {
                'total': 10000,  # TODO: Get from network
                'voted': len(df),
                'percentage': participation
            }
        } 