"""Constants and configurations for the EPC Rating System."""

# EPC rating classes
EPC_RATINGS = ['A', 'B', 'C', 'D', 'E']

# Default class mapping
DEFAULT_CLASS_MAPPING = {i: rating for i, rating in enumerate(EPC_RATINGS)}

# Rating descriptions
RATING_DESCRIPTIONS = {
    'A': "Highly energy efficient with excellent thermal properties and likely renewable energy sources.",
    'B': "Very good energy performance with good insulation and efficient systems.",
    'C': "Good energy efficiency, typical of newer or well-upgraded buildings.",
    'D': "Moderate energy efficiency, common in average buildings.",
    'E': "Below average energy efficiency, improvements recommended."
}

# Color mapping for visualizations
RATING_COLORS = {
    'A': '#00A651',  # Green
    'B': '#50B848',  # Light Green
    'C': '#FFDE17',  # Yellow
    'D': '#FBB040',  # Orange
    'E': '#F15A29'   # Red
}

# Feature importance explanation templates
FEATURE_EXPLANATION = {
    'building_age': {
        'high': "Older buildings typically have poorer energy efficiency without renovations.",
        'low': "Newer buildings often benefit from modern construction standards and materials."
    },
    'floor_area': {
        'high': "Larger floor areas can be harder to heat efficiently without proper zoning.",
        'low': "Smaller spaces are generally easier to heat and maintain energy efficiency."
    },
    'insulation_quality': {
        'high': "Better insulation reduces heat loss dramatically, increasing efficiency.",
        'low': "Poor insulation is a primary cause of energy inefficiency in buildings."
    },
    'energy_consumption': {
        'high': "Higher energy consumption directly relates to poorer EPC ratings.",
        'low': "Lower energy consumption is a key factor in achieving better EPC ratings."
    }
}

# Default sample building data (for UI demo)
SAMPLE_BUILDINGS = [
    {
        'id': 1,
        'name': 'Modern Apartment',
        'building_age': 5,
        'floor_area': 85.0,
        'num_rooms': 3,
        'insulation_quality': 'Excellent',
        'heating_system': 'Electric',
        'glazing_type': 'Triple',
        'roof_insulation': 0.95,
        'wall_insulation': 0.92,
        'energy_consumption': 65.0
    },
    {
        'id': 2,
        'name': 'Renovated Townhouse',
        'building_age': 45,
        'floor_area': 120.0,
        'num_rooms': 5,
        'insulation_quality': 'Good',
        'heating_system': 'Gas',
        'glazing_type': 'Double',
        'roof_insulation': 0.82,
        'wall_insulation': 0.75,
        'energy_consumption': 110.0
    },
    {
        'id': 3,
        'name': 'Older Detached House',
        'building_age': 85,
        'floor_area': 180.0,
        'num_rooms': 7,
        'insulation_quality': 'Average',
        'heating_system': 'Oil',
        'glazing_type': 'Double',
        'roof_insulation': 0.65,
        'wall_insulation': 0.50,
        'energy_consumption': 240.0
    },
    {
        'id': 4,
        'name': 'Eco-Friendly Home',
        'building_age': 3,
        'floor_area': 110.0,
        'num_rooms': 4,
        'insulation_quality': 'Excellent',
        'heating_system': 'Renewable',
        'glazing_type': 'Triple',
        'roof_insulation': 0.98,
        'wall_insulation': 0.96,
        'energy_consumption': 40.0
    },
    {
        'id': 5,
        'name': 'Traditional Cottage',
        'building_age': 120,
        'floor_area': 90.0,
        'num_rooms': 4,
        'insulation_quality': 'Poor',
        'heating_system': 'Electric',
        'glazing_type': 'Single',
        'roof_insulation': 0.35,
        'wall_insulation': 0.30,
        'energy_consumption': 320.0
    }
]
# Feature descriptions for UI
# FEATURE_DESCRIPTIONS = {
# Feature descriptions for UI
# FEATURE_DESCRIPTIONS = {
#     'building_age': 'Age of the building in years',
#     'floor_area': 'Total floor area in square meters',
#     'num_rooms': 'Number of rooms in the building',
#     'insulation_quality':
