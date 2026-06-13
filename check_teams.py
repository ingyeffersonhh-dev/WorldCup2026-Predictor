# -*- coding: utf-8 -*-
import pandas as pd

real_teams = [
    'Mexico', 'South Africa', 'South Korea', 'Czech Republic',
    'Canada', 'Italy', 'Qatar', 'Switzerland',
    'Brazil', 'Morocco', 'Haiti', 'Scotland',
    'United States', 'Paraguay', 'Australia', 'Turkey',
    'Germany', 'Curaçao', 'Ivory Coast', 'Ecuador',
    'Netherlands', 'Japan', 'Tunisia', 'Sweden',
    'Belgium', 'Egypt', 'Iran', 'New Zealand',
    'Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay',
    'France', 'Senegal', 'Norway', 'Chile',
    'Argentina', 'Algeria', 'Austria', 'Jordan',
    'Portugal', 'Colombia', 'Uzbekistan', 'Peru',
    'England', 'Croatia', 'Ghana', 'Panama'
]

df = pd.read_csv('data/processed/elo_history.csv')
teams_in_elo = set(df['team'].unique())

missing = [t for t in real_teams if t not in teams_in_elo]
print("Missing:", missing)
