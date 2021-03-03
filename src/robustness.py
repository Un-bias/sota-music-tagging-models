# coding: utf-8
'''
Deformation codes are borrowed from MUDA
McFee et al., A software framework for musical data augmentation, 2015
https://github.com/bmcfee/muda
'''
import os
import time
import subprocess
import tempfile
import numpy as np
import pandas as pd
import datetime
import tqdm
import csv
import fire
import argparse
import pickle
from sklearn import metrics
import pandas as pd
import librosa
import soundfile as psf

import torch
import torch.nn as nn
from torch.autograd import Variable
from solver import skip_files
from sklearn.preprocessing import LabelBinarizer

import model as Model


TAGS = ['genre---downtempo', 'genre---ambient', 'genre---rock', 'instrument---synthesizer', 'genre---atmospheric', 'genre---indie', 'instrument---electricpiano', 'genre---newage', 'instrument---strings', 'instrument---drums', 'instrument---drummachine', 'genre---techno', 'instrument---guitar', 'genre---alternative', 'genre---easylistening', 'genre---instrumentalpop', 'genre---chillout', 'genre---metal', 'mood/theme---happy', 'genre---lounge', 'genre---reggae', 'genre---popfolk', 'genre---orchestral', 'instrument---acousticguitar', 'genre---poprock', 'instrument---piano', 'genre---trance', 'genre---dance', 'instrument---electricguitar', 'genre---soundtrack', 'genre---house', 'genre---hiphop', 'genre---classical', 'mood/theme---energetic', 'genre---electronic', 'genre---world', 'genre---experimental', 'instrument---violin', 'genre---folk', 'mood/theme---emotional', 'instrument---voice', 'instrument---keyboard', 'genre---pop', 'instrument---bass', 'instrument---computer', 'mood/theme---film', 'genre---triphop', 'genre---jazz', 'genre---funk', 'mood/theme---relaxing']
genres_tags = ['ABSTRACTRO', 'ABSTRACT_BEATS', 'ABSTRACT_HIP_HOP', 'ACID_HOUSE', 'ACID_JAZZ', 'ACID_TECHNO', 'ACOUSTIC_POP', 'ADELAIDE_INDIE', 'ADULT_STANDARDS', 'AFRICAN_ELECTRONIC', 'AFROBEAT', 'AFROPOP', 'AFRO_DANCEHALL', 'AFRO_HOUSE', 'ALABAMA_INDIE', 'ALABAMA_RAP', 'ALASKA_INDIE', 'ALBANIAN_HIP_HOP', 'ALBANIAN_POP', 'ALBERTA_COUNTRY', 'ALBERTA_HIP_HOP', 'ALBUM_ROCK', 'ALBUQUERQUE_INDIE', 'ALTERNATIVE_AMERICANA', 'ALTERNATIVE_COUNTRY', 'ALTERNATIVE_DANCE', 'ALTERNATIVE_EMO', 'ALTERNATIVE_HIP_HOP', 'ALTERNATIVE_METAL', 'ALTERNATIVE_POP', 'ALTERNATIVE_POP_ROCK', 'ALTERNATIVE_RB', 'ALTERNATIVE_ROCK', 'ALTERNATIVE_ROOTS_ROCK', 'AMBEAT', 'AMBIENT', 'AMBIENT_FOLK', 'AMBIENT_IDM', 'AMBIENT_TECHNO', 'AMERICAN_SHOEGAZE', 'ANN_ARBOR_INDIE', 'ANTHEM_EMO', 'ANTHEM_WORSHIP', 'ANTIFOLK', 'ARABIC_HIP_HOP', 'ARGENTINE_HIP_HOP', 'ARGENTINE_INDIE', 'ARGENTINE_ROCK', 'ARKANSAS_COUNTRY', 'ART_POP', 'ART_ROCK', 'ASBURY_PARK_INDIE', 'ASHEVILLE_INDIE', 'ATLANTA_INDIE', 'ATL_HIP_HOP', 'ATL_TRAP', 'AUCKLAND_INDIE', 'AUSSIETRONICA', 'AUSSIE_EMO', 'AUSTINDIE', 'AUSTRALIAN_ALTERNATIVE_POP', 'AUSTRALIAN_ALTERNATIVE_ROCK', 'AUSTRALIAN_COUNTRY', 'AUSTRALIAN_DANCE', 'AUSTRALIAN_ELECTROPOP', 'AUSTRALIAN_GARAGE_PUNK', 'AUSTRALIAN_HIP_HOP', 'AUSTRALIAN_HOUSE', 'AUSTRALIAN_INDIE', 'AUSTRALIAN_INDIE_ROCK', 'AUSTRALIAN_POP', 'AUSTRALIAN_PSYCH', 'AUSTRALIAN_RB', 'AUSTRALIAN_REGGAE_FUSION', 'AUSTRALIAN_SHOEGAZE', 'AUSTRALIAN_SINGERSONGWRITER', 'AUSTRALIAN_TALENT_SHOW', 'AUSTRALIAN_TRAP', 'AUSTRALIAN_UNDERGROUND_HIP_HOP', 'AVANTGARDE', 'AVANTGARDE_JAZZ', 'AZONTO', 'AZONTOBEATS', 'A_CAPPELLA', 'BACHATA', 'BAILE_POP', 'BALEARIC', 'BALTIMORE_HIP_HOP', 'BALTIMORE_INDIE', 'BANDA', 'BANJO', 'BARBADIAN_POP', 'BAROQUE_POP', 'BASSHALL', 'BASSLINE', 'BASS_HOUSE', 'BASS_MUSIC', 'BASS_TRAP', 'BASS_TRIP', 'BATIDA', 'BATON_ROUGE_RAP', 'BATTLE_RAP', 'BAY_AREA_INDIE', 'BBOY', 'BC_UNDERGROUND_HIP_HOP', 'BEBOP', 'BEDROOM_POP', 'BEDROOM_SOUL', 'BELFAST_INDIE', 'BELGIAN_DANCE', 'BELGIAN_EDM', 'BELGIAN_HIP_HOP', 'BELGIAN_INDIE', 'BELGIAN_INDIE_ROCK', 'BELGIAN_MODERN_JAZZ', 'BELGIAN_POP', 'BELGIAN_ROCK', 'BERGEN_INDIE', 'BIG_BAND', 'BIG_BEAT', 'BIG_ROOM', 'BIRMINGHAM_GRIME', 'BIRMINGHAM_HIP_HOP', 'BIRMINGHAM_INDIE', 'BIRMINGHAM_METAL', 'BLUES_ROCK', 'BMORE', 'BOOGALOO', 'BOOM_BAP', 'BOSSA_NOVA', 'BOSSA_NOVA_COVER', 'BOSTON_HIP_HOP', 'BOSTON_INDIE', 'BOSTON_ROCK', 'BOUNCE', 'BOW_POP', 'BOY_BAND', 'BRASS_BAND', 'BRAZILIAN_EDM', 'BRAZILIAN_HIP_HOP', 'BRAZILIAN_HOUSE', 'BRAZILIAN_MODERN_JAZZ', 'BRAZILIAN_PSYCHEDELIC', 'BRAZILIAN_SOUL', 'BREAKBEAT', 'BREGA_FUNK', 'BRIGHTON_INDIE', 'BRILL_BUILDING_POP', 'BRISBANE_INDIE', 'BRISTOL_INDIE', 'BRITISH_ALTERNATIVE_ROCK', 'BRITISH_EXPERIMENTAL', 'BRITISH_FOLK', 'BRITISH_INDIE_ROCK', 'BRITISH_INVASION', 'BRITISH_JAZZ', 'BRITISH_SINGERSONGWRITER', 'BRITISH_SOUL', 'BRITPOP', 'BROADWAY', 'BROKEN_BEAT', 'BROOKLYN_INDIE', 'BROSTEP', 'BUBBLEGUM_DANCE', 'BULGARIAN_INDIE', 'CALI_RAP', 'CALMING_INSTRUMENTAL', 'CAMBRIDGESHIRE_INDIE', 'CANADIAN_CONTEMPORARY_COUNTRY', 'CANADIAN_CONTEMPORARY_RB', 'CANADIAN_COUNTRY', 'CANADIAN_ELECTRONIC', 'CANADIAN_ELECTROPOP', 'CANADIAN_EXPERIMENTAL', 'CANADIAN_FOLK', 'CANADIAN_HIP_HOP', 'CANADIAN_INDIE', 'CANADIAN_LATIN', 'CANADIAN_MODERN_JAZZ', 'CANADIAN_POP', 'CANADIAN_POP_PUNK', 'CANADIAN_POSTHARDCORE', 'CANADIAN_PUNK', 'CANADIAN_ROCK', 'CANADIAN_SHOEGAZE', 'CANADIAN_SINGERSONGWRITER', 'CANDY_POP', 'CANTAUTOR', 'CAPE_TOWN_INDIE', 'CATSTEP', 'CCM', 'CDMX_INDIE', 'CEDM', 'CELTIC_ROCK', 'CHAMBER_POP', 'CHAMBER_PSYCH', 'CHAMPETA', 'CHANNEL_ISLANDS_INDIE', 'CHANNEL_POP', 'CHARLOTTESVILLE_INDIE', 'CHARLOTTE_NC_INDIE', 'CHICAGO_DRILL', 'CHICAGO_HOUSE', 'CHICAGO_INDIE', 'CHICAGO_PUNK', 'CHICAGO_RAP', 'CHILEAN_INDIE', 'CHILLHOP', 'CHILLSTEP', 'CHILLWAVE', 'CHINESE_HIP_HOP', 'CHINESE_INDIE', 'CHRISTCHURCH_INDIE', 'CHRISTIAN_ALTERNATIVE_ROCK', 'CHRISTIAN_HIP_HOP', 'CHRISTIAN_INDIE', 'CHRISTIAN_MUSIC', 'CHRISTIAN_POP', 'CHRISTIAN_TRAP', 'CHRISTLICHER_RAP', 'CIRCUIT', 'CLASSIC_COUNTRY_POP', 'CLASSIC_ITALIAN_POP', 'CLASSIC_ROCK', 'CLASSIC_SWEDISH_POP', 'CLASSIFY', 'COLLAGE_POP', 'COLOGNE_INDIE', 'COLOMBIAN_HIP_HOP', 'COLOMBIAN_INDIE', 'COLOMBIAN_POP', 'COLUMBUS_OHIO_INDIE', 'COMPLEXTRO', 'COMPOSITIONAL_AMBIENT', 'CONNECTICUT_INDIE', 'CONSCIOUS_HIP_HOP', 'CONTEMPORARY_COUNTRY', 'CONTEMPORARY_JAZZ', 'CONTEMPORARY_POSTBOP', 'COOL_JAZZ', 'COUNTRY', 'COUNTRY_DAWN', 'COUNTRY_POP', 'COUNTRY_RAP', 'COUNTRY_ROAD', 'COUNTRY_ROCK', 'COVERCHILL', 'CRUNK', 'CUBAN_RUMBA', 'CUMBIA', 'CUMBIA_POP', 'DANCEHALL', 'DANCEPUNK', 'DANCE_POP', 'DANCE_ROCK', 'DANISH_ALTERNATIVE_ROCK', 'DANISH_ELECTRONIC', 'DANISH_ELECTROPOP', 'DANISH_INDIE_POP', 'DANISH_JAZZ', 'DANISH_METAL', 'DANISH_POP', 'DARK_DISCO', 'DARK_JAZZ', 'DARK_POSTPUNK', 'DARK_TECHNO', 'DARK_TRAP', 'DC_INDIE', 'DEEP_BIG_ROOM', 'DEEP_DISCO_HOUSE', 'DEEP_DUBSTEP', 'DEEP_EURO_HOUSE', 'DEEP_GERMAN_HIP_HOP', 'DEEP_GROOVE_HOUSE', 'DEEP_HOUSE', 'DEEP_IDM', 'DEEP_LATIN_ALTERNATIVE', 'DEEP_LIQUID_BASS', 'DEEP_MELODIC_EURO_HOUSE', 'DEEP_MINIMAL_TECHNO', 'DEEP_NEW_AMERICANA', 'DEEP_POP_EDM', 'DEEP_POP_RB', 'DEEP_SOUL_HOUSE', 'DEEP_SOUTHERN_TRAP', 'DEEP_TALENT_SHOW', 'DEEP_TECHNO', 'DEEP_TECH_HOUSE', 'DEEP_TROPICAL_HOUSE', 'DEEP_UNDERGROUND_HIP_HOP', 'DEMBOW', 'DENTON_TX_INDIE', 'DENVER_INDIE', 'DERRY_INDIE', 'DESI_HIP_HOP', 'DESI_POP', 'DESTROY_TECHNO', 'DETROIT_HIP_HOP', 'DETROIT_INDIE', 'DETROIT_TECHNO', 'DETROIT_TRAP', 'DEVON_INDIE', 'DFW_RAP', 'DIRTY_SOUTH_RAP', 'DISCO', 'DISCO_HOUSE', 'DIVA_HOUSE', 'DIY_EMO', 'DMV_RAP', 'DOMINICAN_POP', 'DOWNTEMPO', 'DREAMGAZE', 'DREAMO', 'DREAM_POP', 'DRIFT', 'DRILL', 'DRILL_AND_BASS', 'DRONE', 'DRUMFUNK', 'DRUM_AND_BASS', 'DUBLIN_INDIE', 'DUBSTEP', 'DUB_TECHNO', 'DUTCH_CABARET', 'DUTCH_EXPERIMENTAL_ELECTRONIC', 'DUTCH_HIP_HOP', 'DUTCH_HOUSE', 'DUTCH_INDIE', 'DUTCH_JAZZ', 'DUTCH_POP', 'DUTCH_ROCK', 'DUTCH_URBAN', 'EAST_ANGLIA_INDIE', 'EAST_COAST_HIP_HOP', 'EASYCORE', 'EASY_LISTENING', 'EAU_CLAIRE_INDIE', 'ECTOFOLK', 'EDINBURGH_INDIE', 'EDM', 'EDMONTON_INDIE', 'ELECTRA', 'ELECTRIC_BLUES', 'ELECTROCLASH', 'ELECTROFOX', 'ELECTRONICA', 'ELECTRONIC_ROCK', 'ELECTRONIC_TRAP', 'ELECTROPOP', 'ELECTROPOWERPOP', 'ELECTRO_HOUSE', 'ELECTRO_LATINO', 'ELECTRO_SWING', 'EL_PASO_INDIE', 'EMO', 'EMO_RAP', 'ENGLISH_INDIE_ROCK', 'ESCAPE_ROOM', 'ETHERPOP', 'ETHIOJAZZ', 'ETHNOTRONICA', 'EUPHORIC_HARDSTYLE', 'EURODANCE', 'EUROPOP', 'EUROVISION', 'EXPERIMENTAL', 'EXPERIMENTAL_AMBIENT', 'EXPERIMENTAL_ELECTRONIC', 'EXPERIMENTAL_FOLK', 'EXPERIMENTAL_HIP_HOP', 'EXPERIMENTAL_HOUSE', 'EXPERIMENTAL_POP', 'EXPERIMENTAL_PSYCH', 'EXPERIMENTAL_TECHNO', 'FIDGET_HOUSE', 'FILMI', 'FILTER_HOUSE', 'FILTHSTEP', 'FINNISH_EDM', 'FINNISH_ELECTRO', 'FINNISH_INDIE', 'FINNISH_JAZZ', 'FLOAT_HOUSE', 'FLORIDA_RAP', 'FLUXWORK', 'FOCUS', 'FOCUS_TRANCE', 'FOLK', 'FOLKPOP', 'FOLKTRONICA', 'FOLK_BRASILEIRO', 'FOLK_PUNK', 'FOLK_ROCK', 'FOOTWORK', 'FORRO', 'FORT_WORTH_INDIE', 'FOURTH_WORLD', 'FRANCOTON', 'FRANKFURT_ELECTRONIC', 'FREAK_FOLK', 'FREE_IMPROVISATION', 'FREE_JAZZ', 'FRENCH_HIP_HOP', 'FRENCH_INDIETRONICA', 'FRENCH_INDIE_POP', 'FRENCH_JAZZ', 'FRENCH_SHOEGAZE', 'FRENCH_TECHNO', 'FUNK', 'FUNKY_TECH_HOUSE', 'FUNK_CARIOCA', 'FUNK_DAS_ANTIGAS', 'FUNK_METAL', 'FUNK_OSTENTACAO', 'FUNK_ROCK', 'FUTURE_FUNK', 'FUTURE_GARAGE', 'FUTURE_HOUSE', 'GANGSTER_RAP', 'GARAGE_POP', 'GARAGE_PSYCH', 'GARAGE_PUNK', 'GARAGE_ROCK', 'GAUZE_POP', 'GERMAN_CLOUD_RAP', 'GERMAN_DANCE', 'GERMAN_HIP_HOP', 'GERMAN_HOUSE', 'GERMAN_INDIE', 'GERMAN_INDIE_FOLK', 'GERMAN_INDIE_ROCK', 'GERMAN_JAZZ', 'GERMAN_METAL', 'GERMAN_POP', 'GERMAN_ROCK', 'GERMAN_TECHNO', 'GHANAIAN_HIP_HOP', 'GIRL_GROUP', 'GLAM_METAL', 'GLAM_ROCK', 'GLASGOW_INDIE', 'GLITCH', 'GOSPEL', 'GOSPEL_RB', 'GOTHENBURG_INDIE', 'GQOM', 'GRAND_RAPIDS_INDIE', 'GRAVE_WAVE', 'GREEK_HOUSE', 'GRIME', 'GRIMEWAVE', 'GROOVE_METAL', 'GROOVE_ROOM', 'GRUNGE', 'G_FUNK', 'HALIFAX_INDIE', 'HAMBURG_ELECTRONIC', 'HAMBURG_HIP_HOP', 'HARDCORE_HIP_HOP', 'HARDCORE_TECHNO', 'HARDSTYLE', 'HARD_ROCK', 'HAWAIIAN_HIP_HOP', 'HEARTLAND_ROCK', 'HINRG', 'HIP_HOP', 'HIP_HOP_QUEBECOIS', 'HIP_HOUSE', 'HIP_POP', 'HOLLYWOOD', 'HOPEBEAT', 'HORROR_SYNTH', 'HOUSE', 'HOUSTON_INDIE', 'HOUSTON_RAP', 'HYPERPOP', 'HYPHY', 'ICELANDIC_ELECTRONIC', 'ICELANDIC_INDIE', 'ICELANDIC_POP', 'ICELANDIC_ROCK', 'IDOL', 'INDIANA_INDIE', 'INDIECOUSTICA', 'INDIETRONICA', 'INDIE_ANTHEMFOLK', 'INDIE_CAFE_POP', 'INDIE_DEUTSCHRAP', 'INDIE_DREAM_POP', 'INDIE_ELECTRONICA', 'INDIE_ELECTROPOP', 'INDIE_FOLK', 'INDIE_GARAGE_ROCK', 'INDIE_JAZZ', 'INDIE_POP', 'INDIE_POPTIMISM', 'INDIE_POP_RAP', 'INDIE_PSYCHPOP', 'INDIE_PUNK', 'INDIE_QUEBECOIS', 'INDIE_RB', 'INDIE_ROCK', 'INDIE_ROCKISM', 'INDIE_SHOEGAZE', 'INDIE_SOUL', 'INDIE_SURF', 'INDIE_TICO', 'INDONESIAN_EDM', 'INDONESIAN_HIP_HOP', 'INDONESIAN_JAZZ', 'INDONESIAN_POP', 'INDONESIAN_RB', 'INDUSTRIAL', 'INDUSTRIAL_METAL', 'INDY_INDIE', 'INSTRUMENTAL_FUNK', 'INSTRUMENTAL_GRIME', 'INTELLIGENT_DANCE_MUSIC', 'IRANIAN_EXPERIMENTAL', 'IRISH_ELECTRONIC', 'IRISH_HIP_HOP', 'IRISH_INDIE', 'IRISH_INDIE_ROCK', 'IRISH_POP', 'IRISH_ROCK', 'IRISH_SINGERSONGWRITER', 'ISLE_OF_WIGHT_INDIE', 'ISRAELI_HIP_HOP', 'ISRAELI_JAZZ', 'ISRAELI_POP', 'ITALIAN_ALTERNATIVE', 'ITALIAN_ARENA_POP', 'ITALIAN_HIP_HOP', 'ITALIAN_JAZZ', 'ITALIAN_POP', 'ITALIAN_TECHNO', 'ITALIAN_TECH_HOUSE', 'ITALO_DANCE', 'JACKSONVILLE_INDIE', 'JAMBIENT', 'JAMTRONICA', 'JAPANESE_CITY_POP', 'JAPANESE_EXPERIMENTAL', 'JAPANESE_JAZZ', 'JAPANESE_RB', 'JAZZ', 'JAZZTRONICA', 'JAZZ_BOOM_BAP', 'JAZZ_BRASS', 'JAZZ_CUBANO', 'JAZZ_DOUBLE_BASS', 'JAZZ_DRUMS', 'JAZZ_ELECTRIC_BASS', 'JAZZ_FUNK', 'JAZZ_FUSION', 'JAZZ_GUITAR', 'JAZZ_MEXICANO', 'JAZZ_PIANO', 'JAZZ_QUARTET', 'JAZZ_RAP', 'JAZZ_SAXOPHONE', 'JAZZ_TRIO', 'JAZZ_TRUMPET', 'JAZZ_VIOLIN', 'JDANCE', 'JPOP', 'JRAP', 'JUMP_UP', 'KC_INDIE', 'KENTUCKY_INDIE', 'KENT_INDIE', 'KHOP', 'KINDIE', 'KINGSTON_ON_INDIE', 'KOREAN_POP', 'KOREAN_RB', 'KOSOVAN_POP', 'KPOP', 'KPOP_BOY_GROUP', 'KPOP_GIRL_GROUP', 'KWAITO_HOUSE', 'LATIN', 'LATINTRONICA', 'LATIN_ALTERNATIVE', 'LATIN_ARENA_POP', 'LATIN_HIP_HOP', 'LATIN_JAZZ', 'LATIN_POP', 'LATIN_ROCK', 'LATIN_TALENT_SHOW', 'LATIN_TECH_HOUSE', 'LATIN_VIRAL_POP', 'LA_INDIE', 'LA_POP', 'LEEDS_INDIE', 'LEICESTER_INDIE', 'LEIPZIG_ELECTRONIC', 'LGBTQ_HIP_HOP', 'LILITH', 'LIQUID_FUNK', 'LITHUANIAN_ELECTRONIC', 'LIVERPOOL_INDIE', 'LOFI_BEATS', 'LOFI_HOUSE', 'LONDON_INDIE', 'LONDON_ON_INDIE', 'LONDON_RAP', 'LOUISVILLE_INDIE', 'LOUNGE', 'MALAYSIAN_POP', 'MANCHESTER_HIP_HOP', 'MANCHESTER_INDIE', 'MANDIBLE', 'MANITOBA_INDIE', 'MASHUP', 'MELBOURNE_INDIE', 'MELLOW_GOLD', 'MELODIC_HARDCORE', 'MELODIC_METALCORE', 'MELODIC_RAP', 'MELODIPOP', 'MEME_RAP', 'MEMPHIS_HIP_HOP', 'MEMPHIS_INDIE', 'MERENGUE', 'METAL', 'METALCORE', 'METROPOPOLIS', 'MEXICAN_INDIE', 'MEXICAN_POP', 'MIAMI_HIP_HOP', 'MIAMI_INDIE', 'MICHIGAN_INDIE', 'MICROHOUSE', 'MILAN_INDIE', 'MILWAUKEE_INDIE', 'MINIMAL_DUB', 'MINIMAL_DUBSTEP', 'MINIMAL_TECHNO', 'MINIMAL_TECH_HOUSE', 'MINNEAPOLIS_INDIE', 'MINNEAPOLIS_SOUND', 'MINNESOTA_HIP_HOP', 'MODERN_ALTERNATIVE_ROCK', 'MODERN_BLUES', 'MODERN_BLUES_ROCK', 'MODERN_BOLLYWOOD', 'MODERN_COUNTRY_ROCK', 'MODERN_HARD_ROCK', 'MODERN_REGGAE', 'MODERN_ROCK', 'MODERN_SALSA', 'MONTREAL_INDIE', 'MOOMBAHTON', 'MOROCCAN_POP', 'MOVIE_TUNES', 'MPB', 'MUNICH_ELECTRONIC', 'MUNICH_INDIE', 'MUSICA_CANARIA', 'MUSIQUE_CONCRETE', 'NASHVILLE_INDIE', 'NASHVILLE_SINGERSONGWRITER', 'NASHVILLE_SOUND', 'NATIVE_AMERICAN_HIP_HOP', 'NC_HIP_HOP', 'NEOCLASSICAL', 'NEON_POP_PUNK', 'NEOPSYCHEDELIC', 'NEOROCKABILLY', 'NEOSINGERSONGWRITER', 'NEOTRADITIONAL_COUNTRY', 'NEO_MELLOW', 'NEO_RB', 'NEO_SOUL', 'NEUROFUNK', 'NEWCASTLE_INDIE', 'NEWCASTLE_NSW_INDIE', 'NEWFOUNDLAND_INDIE', 'NEW_AMERICANA', 'NEW_FRENCH_TOUCH', 'NEW_ISOLATIONISM', 'NEW_JACK_SWING', 'NEW_JERSEY_INDIE', 'NEW_JERSEY_RAP', 'NEW_ORLEANS_FUNK', 'NEW_ORLEANS_JAZZ', 'NEW_ORLEANS_RAP', 'NEW_RAVE', 'NEW_WAVE_POP', 'NIGERIAN_HIP_HOP', 'NIGERIAN_POP', 'NINJA', 'NOISE_POP', 'NORDIC_HOUSE', 'NORTENO', 'NORTH_EAST_ENGLAND_INDIE', 'NORWEGIAN_INDIE', 'NORWEGIAN_JAZZ', 'NORWEGIAN_POP', 'NORWEGIAN_TECHNO', 'NOTTINGHAM_INDIE', 'NOVA_MPB', 'NUMETALCORE', 'NU_AGE', 'NU_DISCO', 'NU_GAZE', 'NU_JAZZ', 'NU_METAL', 'NYC_POP', 'NYC_RAP', 'NZ_ELECTRONIC', 'NZ_HIP_HOP', 'NZ_POP', 'OAKLAND_INDIE', 'OKC_INDIE', 'OLYMPIA_WA_INDIE', 'ONTARIO_INDIE', 'ORGANIC_ELECTRONIC', 'ORLANDO_INDIE', 'OSLO_INDIE', 'OTACORE', 'OTTAWA_INDIE', 'OUTLAW_COUNTRY', 'OUTSIDER_HOUSE', 'OXFORD_INDIE', 'PAGODE', 'PANAMANIAN_POP', 'PARTYSCHLAGER', 'PEI_INDIE', 'PERMANENT_WAVE', 'PERREO', 'PERTH_INDIE', 'PHILLY_INDIE', 'PHILLY_RAP', 'PHONK', 'PIANO_ROCK', 'PINOY_INDIE', 'PITTSBURGH_INDIE', 'PITTSBURGH_RAP', 'PIXIE', 'POLISH_ELECTRONICA', 'POLISH_JAZZ', 'POP', 'POPPING', 'POP_ARGENTINO', 'POP_CATRACHO', 'POP_EDM', 'POP_EMO', 'POP_FOLK', 'POP_HOUSE', 'POP_NACIONAL', 'POP_PUNK', 'POP_QUEBECOIS', 'POP_RAP', 'POP_REGGAETON', 'POP_ROCK', 'POP_URBAINE', 'PORTLAND_HIP_HOP', 'PORTLAND_INDIE', 'PORTSMOUTH_INDIE', 'PORTUGUESE_CONTEMPORARY_CLASSICAL', 'PORTUGUESE_JAZZ', 'POSTGRUNGE', 'POSTHARDCORE', 'POSTSCREAMO', 'POSTTEEN_POP', 'PROGRESSIVE_ELECTRO_HOUSE', 'PROGRESSIVE_HOUSE', 'PROGRESSIVE_JAZZ_FUSION', 'PROGRESSIVE_METAL', 'PROGRESSIVE_POSTHARDCORE', 'PROGRESSIVE_TRANCE', 'PROGRESSIVE_TRANCE_HOUSE', 'PSYCHEDELIC_ROCK', 'PUERTO_RICAN_INDIE', 'PUERTO_RICAN_POP', 'PUNK', 'QUEBEC_INDIE', 'RAP', 'RAP_CATALAN', 'RAP_CONSCIENT', 'RAP_DOMINICANO', 'RAP_KREYOL', 'RAP_LATINA', 'RAP_MARSEILLE', 'RAP_METAL', 'RAWSTYLE', 'RAW_TECHNO', 'RB', 'RB_BRASILEIRO', 'RB_EN_ESPANOL', 'REBEL_BLUES', 'REGGAETON', 'REGGAETON_CHILENO', 'REGGAETON_FLOW', 'REGGAE_EN_ESPANOL', 'REGGAE_FUSION', 'REGGAE_ROCK', 'REGIONAL_MEXICAN', 'REGIONAL_MEXICAN_POP', 'RETRO_SOUL', 'ROCHESTER_MN_INDIE', 'ROCK', 'ROCKABILLY', 'ROCK_EN_ESPANOL', 'ROMANIAN_POP', 'ROOTS_AMERICANA', 'RVA_INDIE', 'SAN_DIEGO_INDIE', 'SAN_DIEGO_RAP', 'SCANDINAVIAN_RB', 'SCANDIPOP', 'SCOTTISH_ELECTRONIC', 'SCOTTISH_HIP_HOP', 'SCOTTISH_INDIE', 'SCOTTISH_INDIE_ROCK', 'SCOTTISH_ROCK', 'SCREAMO', 'SCREAM_RAP', 'SEATTLE_INDIE', 'SERTANEJO', 'SERTANEJO_POP', 'SERTANEJO_UNIVERSITARIO', 'SHIMMER_POP', 'SHIMMER_PSYCH', 'SHIVER_POP', 'SHOEGAZE', 'SHOW_TUNES', 'SINGAPOREAN_POP', 'SKATE_PUNK', 'SKWEEE', 'SKY_ROOM', 'SLAYER', 'SMALL_ROOM', 'SMOOTH_JAZZ', 'SMOOTH_SAXOPHONE', 'SOCAL_POP_PUNK', 'SOCIAL_MEDIA_POP', 'SOFT_ROCK', 'SOUL', 'SOUND_ART', 'SOUTHAMPTON_INDIE', 'SOUTHERN_HIP_HOP', 'SOUTH_AFRICAN_ALTERNATIVE', 'SOUTH_AFRICAN_HIP_HOP', 'SOUTH_AFRICAN_POP', 'SOUTH_AFRICAN_ROCK', 'SPANISH_HIP_HOP', 'SPANISH_INDIE_POP', 'SPANISH_NOISE_POP', 'SPANISH_POP', 'SPEED_GARAGE', 'STOCKHOLM_INDIE', 'STOMP_AND_HOLLER', 'SUBSTEP', 'SWEDISH_ALTERNATIVE_ROCK', 'SWEDISH_ELECTRONIC', 'SWEDISH_ELECTROPOP', 'SWEDISH_GANGSTA_RAP', 'SWEDISH_HIP_HOP', 'SWEDISH_IDOL_POP', 'SWEDISH_INDIE_ROCK', 'SWEDISH_JAZZ', 'SWEDISH_POP', 'SWEDISH_REGGAE', 'SWEDISH_SINGERSONGWRITER', 'SWEDISH_SOUL', 'SWEDISH_TECHNO', 'SWEDISH_TROPICAL_HOUSE', 'SWEDISH_URBAN', 'SWING', 'SWISS_INDIE', 'SYDNEY_INDIE', 'TALENT_SHOW', 'TECHNO', 'TECH_HOUSE', 'TEEN_POP', 'TEXAS_COUNTRY', 'TIJUANA_ELECTRONIC', 'TORONTO_INDIE', 'TORONTO_RAP', 'TRANCECORE', 'TRANSPOP', 'TRAP', 'TRAPRUN', 'TRAP_ARGENTINO', 'TRAP_BRASILEIRO', 'TRAP_CATALA', 'TRAP_CHILENO', 'TRAP_ESPANOL', 'TRAP_FRANCAIS', 'TRAP_LATINO', 'TRAP_QUEEN', 'TRAP_SOUL', 'TRIBAL_HOUSE', 'TROPICAL', 'TROPICAL_HOUSE', 'TURKISH_HIP_HOP', 'TURKISH_JAZZ', 'TURKISH_POP', 'UAE_INDIE', 'UKRAINIAN_POP', 'UK_ALTERNATIVE_HIP_HOP', 'UK_ALTERNATIVE_POP', 'UK_AMERICANA', 'UK_CONTEMPORARY_RB', 'UK_DANCE', 'UK_DANCEHALL', 'UK_DNB', 'UK_DRILL', 'UK_EXPERIMENTAL_ELECTRONIC', 'UK_FUNKY', 'UK_HIP_HOP', 'UK_HOUSE', 'UK_POP', 'UK_TECH_HOUSE', 'UNDERGROUND_HIP_HOP', 'UPLIFTING_TRANCE', 'URBAN_CONTEMPORARY', 'UTAH_INDIE', 'VANCOUVER_INDIE', 'VAPOR_HOUSE', 'VAPOR_POP', 'VAPOR_SOUL', 'VAPOR_TRAP', 'VAPOR_TWITCH', 'VENEZUELAN_HIP_HOP', 'VERACRUZ_INDIE', 'VERMONT_INDIE', 'VICTORIA_BC_INDIE', 'VIRAL_POP', 'VIRAL_TRAP', 'VOCAL_HOUSE', 'VOCAL_JAZZ', 'WARM_DRONE', 'WAVE', 'WELSH_INDIE', 'WEST_AUSTRALIAN_HIP_HOP', 'WEST_COAST_TRAP', 'WISCONSIN_INDIE', 'WITCH_HOUSE', 'WONKY', 'WORLD_FUSION', 'ZAPSTEP', 'ZIMDANCEHALL']

def read_file(tsv_file):
    tracks = {}
    with open(tsv_file) as fp:
        reader = csv.reader(fp, delimiter='\t')
        next(reader, None)  # skip header
        for row in reader:
            track_id = row[0]
            tracks[track_id] = {
                'path': row[3].replace('.mp3', '.npy'),
                'tags': row[5:],
            }
    return tracks


class Predict(object):
    def __init__(self, config):
        self.model_type = config.model_type
        self.model_load_path = config.model_load_path
        self.dataset = config.dataset
        self.data_path = config.data_path
        self.batch_size = config.batch_size
        self.is_cuda = torch.cuda.is_available()
        self.build_model()
        self.get_dataset()
        self.mod = config.mod
        self.rate = config.rate
        self.PRESETS = {
                        "radio":            ["0.01,1", "-90,-90,-70,-70,-60,-20,0,0", "-5"],
                        "film standard":    ["0.1,0.3", "-90,-90,-70,-64,-43,-37,-31,-31,-21,-21,0,-20", "0", "0", "0.1"],
                        "film light":       ["0.1,0.3", "-90,-90,-70,-64,-53,-47,-41,-41,-21,-21,0,-20", "0", "0", "0.1"],
                        "music standard":   ["0.1,0.3", "-90,-90,-70,-58,-55,-43,-31,-31,-21,-21,0,-20", "0", "0", "0.1"],
                        "music light":      ["0.1,0.3", "-90,-90,-70,-58,-65,-53,-41,-41,-21,-21,0,-11", "0", "0", "0.1"],
                        "speech":           ["0.1,0.3", "-90,-90,-70,-55,-50,-35,-31,-31,-21,-21,0,-20", "0", "0", "0.1"]
                        }
        self.preset_dict = {1: "radio",
                            2: "film standard",
                            3: "film light",
                            4: "music standard",
                            5: "music light",
                            6: "speech"}
    def get_model(self):
        if self.model_type == 'fcn':
            self.input_length = 29 * 16000
            return Model.FCN()
        elif self.model_type == 'musicnn':
            self.input_length = 3 * 16000
            return Model.Musicnn(dataset=self.dataset)
        elif self.model_type == 'crnn':
            self.input_length = 29 * 16000
            return Model.CRNN()
        elif self.model_type == 'sample':
            self.input_length = 59049
            return Model.SampleCNN()
        elif self.model_type == 'se':
            self.input_length = 59049
            return Model.SampleCNNSE()
        elif self.model_type == 'short':
            self.input_length = 59049
            return Model.ShortChunkCNN()
        elif self.model_type == 'short_res':
            self.input_length = 59049
            return Model.ShortChunkCNN_Res()
        elif self.model_type == 'attention':
            self.input_length = 15 * 16000
            return Model.CNNSA()
        elif self.model_type == 'hcnn':
            self.input_length = 5 * 16000
            return Model.HarmonicCNN()
        else:
            print('model_type has to be one of [fcn, musicnn, crnn, sample, se, short, short_res, attention]')

    def build_model(self):
        self.model = self.get_model()

        # cuda
        if self.is_cuda:
            self.model.cuda()

        # load model
        self.load(self.model_load_path)

    def get_dataset(self):
        if self.dataset == 'mtat':
            self.test_list = np.load('./../split/mtat/test.npy')
            self.binary = np.load('./../split/mtat/binary.npy')
        if self.dataset == 'msd':
            test_file = os.path.join('./../split/msd','filtered_list_test.cP')
            test_list = pickle.load(open(test_file,'rb'), encoding='bytes')
            self.test_list = [value for value in test_list if value.decode() not in skip_files]
            id2tag_file = os.path.join('./../split/msd', 'msd_id_to_tag_vector.cP')
            self.id2tag = pickle.load(open(id2tag_file,'rb'), encoding='bytes')
        if self.dataset == 'jamendo':
            test_file = os.path.join('./../split/mtg-jamendo', 'autotagging_top50tags-test.tsv')
            self.file_dict= read_file(test_file)
            self.test_list= list(self.file_dict.keys())
            self.mlb = LabelBinarizer().fit(TAGS)

        if self.dataset == 'jamendo-mood':
            test_file = os.path.join('./../split/mtg-jamendo-mood', 'autotagging_moodtheme-test.tsv')
            self.file_dict= read_file(test_file)
            self.test_list= list(self.file_dict.keys())
            self.mlb = LabelBinarizer().fit(TAGS)

        if self.dataset == 'genres':
            test_file = os.path.join('./../split/genres', 'test.tsv')
            self.file_dict= read_file(test_file)
            self.test_list= list(self.file_dict.keys())
            self.mlb = LabelBinarizer().fit(genres_tags)

            
    def load(self, filename):
        S = torch.load(filename)
        self.model.load_state_dict(S)

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def get_tensor(self, fn):
        # load audio
        if self.dataset == 'mtat':
            npy_path = os.path.join(self.data_path, 'mtat', 'npy', fn.split('/')[1][:-3]) + 'npy'
        elif self.dataset == 'msd':
            msid = fn.decode()
            filename = '{}/{}/{}/{}.npy'.format(msid[2], msid[3], msid[4], msid)
            npy_path = os.path.join(self.data_path, filename)
        elif self.dataset in ['jamendo', 'jamendo-mood', 'genres']:
            filename = self.file_dict[fn]['path']
            npy_path = os.path.join(self.data_path, filename)
        raw = np.load(npy_path, mmap_mode='r')
        raw = self.modify(raw, self.rate, self.mod)

        # split chunk
        length = len(raw)
        hop = (length - self.input_length) // self.batch_size
        x = torch.zeros(self.batch_size, self.input_length)
        for i in range(self.batch_size):
            x[i] = torch.Tensor(raw[i*hop:i*hop+self.input_length]).unsqueeze(0)
        return x

    def modify(self, x, mod_rate, mod_type):
        if mod_type == 'time_stretch':
            return self.time_stretch(x, mod_rate)
        elif mod_type == 'pitch_shift':
            return self.pitch_shift(x, mod_rate)
        elif mod_type == 'dynamic_range':
            return self.dynamic_range_compression(x, mod_rate)
        elif mod_type == 'white_noise':
            return self.white_noise(x, mod_rate)
        else:
            print('choose from [time_stretch, pitch_shift, dynamic_range, white_noise]')

    def time_stretch(self, x, rate):
        '''
        [2 ** (-.5), 2 ** (.5)]
        '''
        return librosa.effects.time_stretch(x, rate)

    def pitch_shift(self, x, rate):
        '''
        [-1, 1]
        '''
        return librosa.effects.pitch_shift(x, 16000, rate)

    def dynamic_range_compression(self, x, rate):
        '''
        [4, 6]
        Music standard & Speech
        '''
        return self.sox(x, 16000, "compand", *self.PRESETS[self.preset_dict[rate]])

    @staticmethod
    def sox(x, fs, *args):
        assert fs > 0

        fdesc, infile = tempfile.mkstemp(suffix=".wav")
        os.close(fdesc)
        fdesc, outfile = tempfile.mkstemp(suffix=".wav")
        os.close(fdesc)

        psf.write(infile, x, fs)

        try:
            arguments = ["sox", infile, outfile, "-q"]
            arguments.extend(args)

            subprocess.check_call(arguments)

            x_out, fs = psf.read(outfile)
            x_out = x_out.T
            if x.ndim == 1:
                x_out = librosa.to_mono(x_out)

        finally:
            os.unlink(infile)
            os.unlink(outfile)

        return x_out

    def white_noise(self, x, rate):
        '''
        [0.1, 0.4]
        '''
        n_frames = len(x)
        noise_white = np.random.RandomState().randn(n_frames)
        noise_fft = np.fft.rfft(noise_white)
        values = np.linspace(1, n_frames * 0.5 + 1, n_frames // 2 + 1)
        colored_filter = np.linspace(1, n_frames / 2 + 1, n_frames // 2 + 1) ** 0
        noise_filtered = noise_fft * colored_filter
        noise = librosa.util.normalize(np.fft.irfft(noise_filtered)) * (x.max())
        if len(noise) < len(x):
            x = x[:len(noise)]
        return (1 - rate) * x + (noise * rate)

    def get_auc(self, est_array, gt_array):
        roc_aucs  = metrics.roc_auc_score(gt_array, est_array, average='macro')
        pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
        return roc_aucs, pr_aucs

    def test(self):
        roc_auc, pr_auc, loss = self.get_test_score()
        print('loss: %.4f' % loss)
        print('roc_auc: %.4f' % roc_auc)
        print('pr_auc: %.4f' % pr_auc)

    def get_test_score(self):
        self.model = self.model.eval()
        est_array = []
        gt_array = []
        losses = []
        reconst_loss = nn.BCELoss()
        for line in tqdm.tqdm(self.test_list):
            if self.dataset == 'mtat':
                ix, fn = line.split('\t')
            elif self.dataset == 'msd':
                fn = line
                if fn.decode() in skip_files:
                    continue
            elif self.dataset in ['jamendo','jamendo-mood','genres']:
                fn = line

            # load and split
            x = self.get_tensor(fn)

            # ground truth
            if self.dataset == 'mtat':
                ground_truth = self.binary[int(ix)]
            elif self.dataset == 'msd':
                ground_truth = self.id2tag[fn].flatten()
            elif self.dataset in ['jamendo', 'jamendo-mood', 'genres']:
                ground_truth = np.sum(self.mlb.transform(self.file_dict[fn]['tags']), axis=0)

            # forward
            x = self.to_var(x)
            y = torch.tensor([ground_truth.astype('float32') for i in range(self.batch_size)]).cuda()
            out = self.model(x)
            loss = reconst_loss(out, y)
            losses.append(float(loss.data))
            out = out.detach().cpu()

            # estimate
            estimated = np.array(out).mean(axis=0)
            est_array.append(estimated)
            gt_array.append(ground_truth)

        est_array, gt_array = np.array(est_array), np.array(gt_array)
        loss = np.mean(losses)

        roc_auc, pr_auc = self.get_auc(est_array, gt_array)
        return roc_auc, pr_auc, loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='genres', choices=['genres', 'mtat', 'msd', 'jamendo','jamendo-mood'])
    parser.add_argument('--model_type', type=str, default='fcn',
                        choices=['fcn', 'musicnn', 'crnn', 'sample', 'se', 'short', 'short_res', 'attention', 'hcnn'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--model_load_path', type=str, default='.')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--mod', type=str, default='time_stretch')
    parser.add_argument('--rate', type=float, default=0)

    config = parser.parse_args()

    p = Predict(config)
    p.test()






