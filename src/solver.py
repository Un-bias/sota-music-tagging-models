# coding: utf-8
import pickle
import os
import time
import numpy as np
import pandas as pd
from sklearn import metrics
import datetime
import csv
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelBinarizer
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import model as Model


skip_files = set(['TRAIISZ128F42684BB', 'TRAONEQ128F42A8AB7', 'TRADRNH128E0784511', 'TRBGHEU128F92D778F',
                 'TRCHYIF128F1464CE7', 'TRCVDKQ128E0790C86', 'TREWVFM128F146816E', 'TREQRIV128F1468B08',
                 'TREUVBN128F1468AC9', 'TRDKNBI128F14682B0', 'TRFWOAG128F14B12CB', 'TRFIYAF128F14688A6',
                 'TRGYAEZ128F14A473F', 'TRIXPRK128F1468472', 'TRAQKCW128F9352A52', 'TRLAWQU128F1468AC8',
                 'TRMSPLW128F14A544A', 'TRLNGQT128F1468261', 'TROTUWC128F1468AB4', 'TRNDAXE128F934C50E',
                 'TRNHIBI128EF35F57D', 'TRMOREL128F1468AC4',  'TRPNFAG128F146825F', 'TRIXPOY128F14A46C7',
                 'TROCQVE128F1468AC6', 'TRPCXJI128F14688A8', 'TRQKRKL128F1468AAE', 'TRPKNDC128F145998B',
                 'TRRUHEH128F1468AAD', 'TRLUSKX128F14A4E50', 'TRMIRQA128F92F11F1', 'TRSRUXF128F1468784',
                 'TRTNQKQ128F931C74D',  'TRTTUYE128F4244068', 'TRUQZKD128F1468243', 'TRUINWL128F1468258',
                 'TRVRHOY128F14680BC', 'TRWVEYR128F1458A6F', 'TRVLISA128F1468960', 'TRYDUYU128F92F6BE0',
                 'TRYOLFS128F9308346', 'TRMVCVS128F1468256', 'TRZSPHR128F1468AAC', 'TRXBJBW128F92EBD96',
                 'TRYPGJX128F1468479', 'TRYNNNZ128F1468994', 'TRVDOVF128F92DC7F3', 'TRWUHZQ128F1451979',
                 'TRXMAVV128F146825C', 'TRYNMEX128F14A401D', 'TREGWSL128F92C9D42', 'TRJKZDA12903CFBA43',
                  'TRBGJIZ128F92E42BC', 'TRVWNOH128E0788B78', 'TRCGBRK128F146A901'])

TAGS = ['genre---downtempo', 'genre---ambient', 'genre---rock', 'instrument---synthesizer', 'genre---atmospheric', 'genre---indie', 'instrument---electricpiano', 'genre---newage', 'instrument---strings', 'instrument---drums', 'instrument---drummachine', 'genre---techno', 'instrument---guitar', 'genre---alternative', 'genre---easylistening', 'genre---instrumentalpop', 'genre---chillout', 'genre---metal', 'mood/theme---happy', 'genre---lounge', 'genre---reggae', 'genre---popfolk', 'genre---orchestral', 'instrument---acousticguitar', 'genre---poprock', 'instrument---piano', 'genre---trance', 'genre---dance', 'instrument---electricguitar', 'genre---soundtrack', 'genre---house', 'genre---hiphop', 'genre---classical', 'mood/theme---energetic', 'genre---electronic', 'genre---world', 'genre---experimental', 'instrument---violin', 'genre---folk', 'mood/theme---emotional', 'instrument---voice', 'instrument---keyboard', 'genre---pop', 'instrument---bass', 'instrument---computer', 'mood/theme---film', 'genre---triphop', 'genre---jazz', 'genre---funk', 'mood/theme---relaxing']
TAGS_MOOD = ["mood/theme---action", "mood/theme---adventure", "mood/theme---advertising", "mood/theme---background", "mood/theme---ballad", "mood/theme---calm", "mood/theme---children", "mood/theme---christmas", "mood/theme---commercial", "mood/theme---cool", "mood/theme---corporate", "mood/theme---dark", "mood/theme---deep", "mood/theme---documentary", "mood/theme---drama", "mood/theme---dramatic", "mood/theme---dream", "mood/theme---emotional", "mood/theme---energetic", "mood/theme---epic", "mood/theme---fast", "mood/theme---film", "mood/theme---fun", "mood/theme---funny", "mood/theme---game", "mood/theme---groovy", "mood/theme---happy", "mood/theme---heavy", "mood/theme---holiday", "mood/theme---hopeful", "mood/theme---inspiring", "mood/theme---love", "mood/theme---meditative", "mood/theme---melancholic", "mood/theme---melodic", "mood/theme---motivational", "mood/theme---movie", "mood/theme---nature", "mood/theme---party", "mood/theme---positive", "mood/theme---powerful", "mood/theme---relaxing", "mood/theme---retro", "mood/theme---romantic", "mood/theme---sad", "mood/theme---sexy", "mood/theme---slow", "mood/theme---soft", "mood/theme---soundscape", "mood/theme---space", "mood/theme---sport", "mood/theme---summer", "mood/theme---trailer", "mood/theme---travel", "mood/theme---upbeat", "mood/theme---uplifting"]
genres_tags = ['ABSTRACTRO', 'ABSTRACT_BEATS', 'ABSTRACT_HIP_HOP', 'ACID_HOUSE', 'ACID_JAZZ', 'ACID_TECHNO', 'ACOUSTIC_POP', 'ADELAIDE_INDIE', 'ADULT_STANDARDS', 'AFRICAN_ELECTRONIC', 'AFROBEAT', 'AFROPOP', 'AFRO_DANCEHALL', 'AFRO_HOUSE', 'ALABAMA_INDIE', 'ALABAMA_RAP', 'ALASKA_INDIE', 'ALBANIAN_HIP_HOP', 'ALBANIAN_POP', 'ALBERTA_COUNTRY', 'ALBERTA_HIP_HOP', 'ALBUM_ROCK', 'ALBUQUERQUE_INDIE', 'ALTERNATIVE_AMERICANA', 'ALTERNATIVE_COUNTRY', 'ALTERNATIVE_DANCE', 'ALTERNATIVE_EMO', 'ALTERNATIVE_HIP_HOP', 'ALTERNATIVE_METAL', 'ALTERNATIVE_POP', 'ALTERNATIVE_POP_ROCK', 'ALTERNATIVE_RB', 'ALTERNATIVE_ROCK', 'ALTERNATIVE_ROOTS_ROCK', 'AMBEAT', 'AMBIENT', 'AMBIENT_FOLK', 'AMBIENT_IDM', 'AMBIENT_TECHNO', 'AMERICAN_SHOEGAZE', 'ANN_ARBOR_INDIE', 'ANTHEM_EMO', 'ANTHEM_WORSHIP', 'ANTIFOLK', 'ARABIC_HIP_HOP', 'ARGENTINE_HIP_HOP', 'ARGENTINE_INDIE', 'ARGENTINE_ROCK', 'ARKANSAS_COUNTRY', 'ART_POP', 'ART_ROCK', 'ASBURY_PARK_INDIE', 'ASHEVILLE_INDIE', 'ATLANTA_INDIE', 'ATL_HIP_HOP', 'ATL_TRAP', 'AUCKLAND_INDIE', 'AUSSIETRONICA', 'AUSSIE_EMO', 'AUSTINDIE', 'AUSTRALIAN_ALTERNATIVE_POP', 'AUSTRALIAN_ALTERNATIVE_ROCK', 'AUSTRALIAN_COUNTRY', 'AUSTRALIAN_DANCE', 'AUSTRALIAN_ELECTROPOP', 'AUSTRALIAN_GARAGE_PUNK', 'AUSTRALIAN_HIP_HOP', 'AUSTRALIAN_HOUSE', 'AUSTRALIAN_INDIE', 'AUSTRALIAN_INDIE_ROCK', 'AUSTRALIAN_POP', 'AUSTRALIAN_PSYCH', 'AUSTRALIAN_RB', 'AUSTRALIAN_REGGAE_FUSION', 'AUSTRALIAN_SHOEGAZE', 'AUSTRALIAN_SINGERSONGWRITER', 'AUSTRALIAN_TALENT_SHOW', 'AUSTRALIAN_TRAP', 'AUSTRALIAN_UNDERGROUND_HIP_HOP', 'AVANTGARDE', 'AVANTGARDE_JAZZ', 'AZONTO', 'AZONTOBEATS', 'A_CAPPELLA', 'BACHATA', 'BAILE_POP', 'BALEARIC', 'BALTIMORE_HIP_HOP', 'BALTIMORE_INDIE', 'BANDA', 'BANJO', 'BARBADIAN_POP', 'BAROQUE_POP', 'BASSHALL', 'BASSLINE', 'BASS_HOUSE', 'BASS_MUSIC', 'BASS_TRAP', 'BASS_TRIP', 'BATIDA', 'BATON_ROUGE_RAP', 'BATTLE_RAP', 'BAY_AREA_INDIE', 'BBOY', 'BC_UNDERGROUND_HIP_HOP', 'BEBOP', 'BEDROOM_POP', 'BEDROOM_SOUL', 'BELFAST_INDIE', 'BELGIAN_DANCE', 'BELGIAN_EDM', 'BELGIAN_HIP_HOP', 'BELGIAN_INDIE', 'BELGIAN_INDIE_ROCK', 'BELGIAN_MODERN_JAZZ', 'BELGIAN_POP', 'BELGIAN_ROCK', 'BERGEN_INDIE', 'BIG_BAND', 'BIG_BEAT', 'BIG_ROOM', 'BIRMINGHAM_GRIME', 'BIRMINGHAM_HIP_HOP', 'BIRMINGHAM_INDIE', 'BIRMINGHAM_METAL', 'BLUES_ROCK', 'BMORE', 'BOOGALOO', 'BOOM_BAP', 'BOSSA_NOVA', 'BOSSA_NOVA_COVER', 'BOSTON_HIP_HOP', 'BOSTON_INDIE', 'BOSTON_ROCK', 'BOUNCE', 'BOW_POP', 'BOY_BAND', 'BRASS_BAND', 'BRAZILIAN_EDM', 'BRAZILIAN_HIP_HOP', 'BRAZILIAN_HOUSE', 'BRAZILIAN_MODERN_JAZZ', 'BRAZILIAN_PSYCHEDELIC', 'BRAZILIAN_SOUL', 'BREAKBEAT', 'BREGA_FUNK', 'BRIGHTON_INDIE', 'BRILL_BUILDING_POP', 'BRISBANE_INDIE', 'BRISTOL_INDIE', 'BRITISH_ALTERNATIVE_ROCK', 'BRITISH_EXPERIMENTAL', 'BRITISH_FOLK', 'BRITISH_INDIE_ROCK', 'BRITISH_INVASION', 'BRITISH_JAZZ', 'BRITISH_SINGERSONGWRITER', 'BRITISH_SOUL', 'BRITPOP', 'BROADWAY', 'BROKEN_BEAT', 'BROOKLYN_INDIE', 'BROSTEP', 'BUBBLEGUM_DANCE', 'BULGARIAN_INDIE', 'CALI_RAP', 'CALMING_INSTRUMENTAL', 'CAMBRIDGESHIRE_INDIE', 'CANADIAN_CONTEMPORARY_COUNTRY', 'CANADIAN_CONTEMPORARY_RB', 'CANADIAN_COUNTRY', 'CANADIAN_ELECTRONIC', 'CANADIAN_ELECTROPOP', 'CANADIAN_EXPERIMENTAL', 'CANADIAN_FOLK', 'CANADIAN_HIP_HOP', 'CANADIAN_INDIE', 'CANADIAN_LATIN', 'CANADIAN_MODERN_JAZZ', 'CANADIAN_POP', 'CANADIAN_POP_PUNK', 'CANADIAN_POSTHARDCORE', 'CANADIAN_PUNK', 'CANADIAN_ROCK', 'CANADIAN_SHOEGAZE', 'CANADIAN_SINGERSONGWRITER', 'CANDY_POP', 'CANTAUTOR', 'CAPE_TOWN_INDIE', 'CATSTEP', 'CCM', 'CDMX_INDIE', 'CEDM', 'CELTIC_ROCK', 'CHAMBER_POP', 'CHAMBER_PSYCH', 'CHAMPETA', 'CHANNEL_ISLANDS_INDIE', 'CHANNEL_POP', 'CHARLOTTESVILLE_INDIE', 'CHARLOTTE_NC_INDIE', 'CHICAGO_DRILL', 'CHICAGO_HOUSE', 'CHICAGO_INDIE', 'CHICAGO_PUNK', 'CHICAGO_RAP', 'CHILEAN_INDIE', 'CHILLHOP', 'CHILLSTEP', 'CHILLWAVE', 'CHINESE_HIP_HOP', 'CHINESE_INDIE', 'CHRISTCHURCH_INDIE', 'CHRISTIAN_ALTERNATIVE_ROCK', 'CHRISTIAN_HIP_HOP', 'CHRISTIAN_INDIE', 'CHRISTIAN_MUSIC', 'CHRISTIAN_POP', 'CHRISTIAN_TRAP', 'CHRISTLICHER_RAP', 'CIRCUIT', 'CLASSIC_COUNTRY_POP', 'CLASSIC_ITALIAN_POP', 'CLASSIC_ROCK', 'CLASSIC_SWEDISH_POP', 'CLASSIFY', 'COLLAGE_POP', 'COLOGNE_INDIE', 'COLOMBIAN_HIP_HOP', 'COLOMBIAN_INDIE', 'COLOMBIAN_POP', 'COLUMBUS_OHIO_INDIE', 'COMPLEXTRO', 'COMPOSITIONAL_AMBIENT', 'CONNECTICUT_INDIE', 'CONSCIOUS_HIP_HOP', 'CONTEMPORARY_COUNTRY', 'CONTEMPORARY_JAZZ', 'CONTEMPORARY_POSTBOP', 'COOL_JAZZ', 'COUNTRY', 'COUNTRY_DAWN', 'COUNTRY_POP', 'COUNTRY_RAP', 'COUNTRY_ROAD', 'COUNTRY_ROCK', 'COVERCHILL', 'CRUNK', 'CUBAN_RUMBA', 'CUMBIA', 'CUMBIA_POP', 'DANCEHALL', 'DANCEPUNK', 'DANCE_POP', 'DANCE_ROCK', 'DANISH_ALTERNATIVE_ROCK', 'DANISH_ELECTRONIC', 'DANISH_ELECTROPOP', 'DANISH_INDIE_POP', 'DANISH_JAZZ', 'DANISH_METAL', 'DANISH_POP', 'DARK_DISCO', 'DARK_JAZZ', 'DARK_POSTPUNK', 'DARK_TECHNO', 'DARK_TRAP', 'DC_INDIE', 'DEEP_BIG_ROOM', 'DEEP_DISCO_HOUSE', 'DEEP_DUBSTEP', 'DEEP_EURO_HOUSE', 'DEEP_GERMAN_HIP_HOP', 'DEEP_GROOVE_HOUSE', 'DEEP_HOUSE', 'DEEP_IDM', 'DEEP_LATIN_ALTERNATIVE', 'DEEP_LIQUID_BASS', 'DEEP_MELODIC_EURO_HOUSE', 'DEEP_MINIMAL_TECHNO', 'DEEP_NEW_AMERICANA', 'DEEP_POP_EDM', 'DEEP_POP_RB', 'DEEP_SOUL_HOUSE', 'DEEP_SOUTHERN_TRAP', 'DEEP_TALENT_SHOW', 'DEEP_TECHNO', 'DEEP_TECH_HOUSE', 'DEEP_TROPICAL_HOUSE', 'DEEP_UNDERGROUND_HIP_HOP', 'DEMBOW', 'DENTON_TX_INDIE', 'DENVER_INDIE', 'DERRY_INDIE', 'DESI_HIP_HOP', 'DESI_POP', 'DESTROY_TECHNO', 'DETROIT_HIP_HOP', 'DETROIT_INDIE', 'DETROIT_TECHNO', 'DETROIT_TRAP', 'DEVON_INDIE', 'DFW_RAP', 'DIRTY_SOUTH_RAP', 'DISCO', 'DISCO_HOUSE', 'DIVA_HOUSE', 'DIY_EMO', 'DMV_RAP', 'DOMINICAN_POP', 'DOWNTEMPO', 'DREAMGAZE', 'DREAMO', 'DREAM_POP', 'DRIFT', 'DRILL', 'DRILL_AND_BASS', 'DRONE', 'DRUMFUNK', 'DRUM_AND_BASS', 'DUBLIN_INDIE', 'DUBSTEP', 'DUB_TECHNO', 'DUTCH_CABARET', 'DUTCH_EXPERIMENTAL_ELECTRONIC', 'DUTCH_HIP_HOP', 'DUTCH_HOUSE', 'DUTCH_INDIE', 'DUTCH_JAZZ', 'DUTCH_POP', 'DUTCH_ROCK', 'DUTCH_URBAN', 'EAST_ANGLIA_INDIE', 'EAST_COAST_HIP_HOP', 'EASYCORE', 'EASY_LISTENING', 'EAU_CLAIRE_INDIE', 'ECTOFOLK', 'EDINBURGH_INDIE', 'EDM', 'EDMONTON_INDIE', 'ELECTRA', 'ELECTRIC_BLUES', 'ELECTROCLASH', 'ELECTROFOX', 'ELECTRONICA', 'ELECTRONIC_ROCK', 'ELECTRONIC_TRAP', 'ELECTROPOP', 'ELECTROPOWERPOP', 'ELECTRO_HOUSE', 'ELECTRO_LATINO', 'ELECTRO_SWING', 'EL_PASO_INDIE', 'EMO', 'EMO_RAP', 'ENGLISH_INDIE_ROCK', 'ESCAPE_ROOM', 'ETHERPOP', 'ETHIOJAZZ', 'ETHNOTRONICA', 'EUPHORIC_HARDSTYLE', 'EURODANCE', 'EUROPOP', 'EUROVISION', 'EXPERIMENTAL', 'EXPERIMENTAL_AMBIENT', 'EXPERIMENTAL_ELECTRONIC', 'EXPERIMENTAL_FOLK', 'EXPERIMENTAL_HIP_HOP', 'EXPERIMENTAL_HOUSE', 'EXPERIMENTAL_POP', 'EXPERIMENTAL_PSYCH', 'EXPERIMENTAL_TECHNO', 'FIDGET_HOUSE', 'FILMI', 'FILTER_HOUSE', 'FILTHSTEP', 'FINNISH_EDM', 'FINNISH_ELECTRO', 'FINNISH_INDIE', 'FINNISH_JAZZ', 'FLOAT_HOUSE', 'FLORIDA_RAP', 'FLUXWORK', 'FOCUS', 'FOCUS_TRANCE', 'FOLK', 'FOLKPOP', 'FOLKTRONICA', 'FOLK_BRASILEIRO', 'FOLK_PUNK', 'FOLK_ROCK', 'FOOTWORK', 'FORRO', 'FORT_WORTH_INDIE', 'FOURTH_WORLD', 'FRANCOTON', 'FRANKFURT_ELECTRONIC', 'FREAK_FOLK', 'FREE_IMPROVISATION', 'FREE_JAZZ', 'FRENCH_HIP_HOP', 'FRENCH_INDIETRONICA', 'FRENCH_INDIE_POP', 'FRENCH_JAZZ', 'FRENCH_SHOEGAZE', 'FRENCH_TECHNO', 'FUNK', 'FUNKY_TECH_HOUSE', 'FUNK_CARIOCA', 'FUNK_DAS_ANTIGAS', 'FUNK_METAL', 'FUNK_OSTENTACAO', 'FUNK_ROCK', 'FUTURE_FUNK', 'FUTURE_GARAGE', 'FUTURE_HOUSE', 'GANGSTER_RAP', 'GARAGE_POP', 'GARAGE_PSYCH', 'GARAGE_PUNK', 'GARAGE_ROCK', 'GAUZE_POP', 'GERMAN_CLOUD_RAP', 'GERMAN_DANCE', 'GERMAN_HIP_HOP', 'GERMAN_HOUSE', 'GERMAN_INDIE', 'GERMAN_INDIE_FOLK', 'GERMAN_INDIE_ROCK', 'GERMAN_JAZZ', 'GERMAN_METAL', 'GERMAN_POP', 'GERMAN_ROCK', 'GERMAN_TECHNO', 'GHANAIAN_HIP_HOP', 'GIRL_GROUP', 'GLAM_METAL', 'GLAM_ROCK', 'GLASGOW_INDIE', 'GLITCH', 'GOSPEL', 'GOSPEL_RB', 'GOTHENBURG_INDIE', 'GQOM', 'GRAND_RAPIDS_INDIE', 'GRAVE_WAVE', 'GREEK_HOUSE', 'GRIME', 'GRIMEWAVE', 'GROOVE_METAL', 'GROOVE_ROOM', 'GRUNGE', 'G_FUNK', 'HALIFAX_INDIE', 'HAMBURG_ELECTRONIC', 'HAMBURG_HIP_HOP', 'HARDCORE_HIP_HOP', 'HARDCORE_TECHNO', 'HARDSTYLE', 'HARD_ROCK', 'HAWAIIAN_HIP_HOP', 'HEARTLAND_ROCK', 'HINRG', 'HIP_HOP', 'HIP_HOP_QUEBECOIS', 'HIP_HOUSE', 'HIP_POP', 'HOLLYWOOD', 'HOPEBEAT', 'HORROR_SYNTH', 'HOUSE', 'HOUSTON_INDIE', 'HOUSTON_RAP', 'HYPERPOP', 'HYPHY', 'ICELANDIC_ELECTRONIC', 'ICELANDIC_INDIE', 'ICELANDIC_POP', 'ICELANDIC_ROCK', 'IDOL', 'INDIANA_INDIE', 'INDIECOUSTICA', 'INDIETRONICA', 'INDIE_ANTHEMFOLK', 'INDIE_CAFE_POP', 'INDIE_DEUTSCHRAP', 'INDIE_DREAM_POP', 'INDIE_ELECTRONICA', 'INDIE_ELECTROPOP', 'INDIE_FOLK', 'INDIE_GARAGE_ROCK', 'INDIE_JAZZ', 'INDIE_POP', 'INDIE_POPTIMISM', 'INDIE_POP_RAP', 'INDIE_PSYCHPOP', 'INDIE_PUNK', 'INDIE_QUEBECOIS', 'INDIE_RB', 'INDIE_ROCK', 'INDIE_ROCKISM', 'INDIE_SHOEGAZE', 'INDIE_SOUL', 'INDIE_SURF', 'INDIE_TICO', 'INDONESIAN_EDM', 'INDONESIAN_HIP_HOP', 'INDONESIAN_JAZZ', 'INDONESIAN_POP', 'INDONESIAN_RB', 'INDUSTRIAL', 'INDUSTRIAL_METAL', 'INDY_INDIE', 'INSTRUMENTAL_FUNK', 'INSTRUMENTAL_GRIME', 'INTELLIGENT_DANCE_MUSIC', 'IRANIAN_EXPERIMENTAL', 'IRISH_ELECTRONIC', 'IRISH_HIP_HOP', 'IRISH_INDIE', 'IRISH_INDIE_ROCK', 'IRISH_POP', 'IRISH_ROCK', 'IRISH_SINGERSONGWRITER', 'ISLE_OF_WIGHT_INDIE', 'ISRAELI_HIP_HOP', 'ISRAELI_JAZZ', 'ISRAELI_POP', 'ITALIAN_ALTERNATIVE', 'ITALIAN_ARENA_POP', 'ITALIAN_HIP_HOP', 'ITALIAN_JAZZ', 'ITALIAN_POP', 'ITALIAN_TECHNO', 'ITALIAN_TECH_HOUSE', 'ITALO_DANCE', 'JACKSONVILLE_INDIE', 'JAMBIENT', 'JAMTRONICA', 'JAPANESE_CITY_POP', 'JAPANESE_EXPERIMENTAL', 'JAPANESE_JAZZ', 'JAPANESE_RB', 'JAZZ', 'JAZZTRONICA', 'JAZZ_BOOM_BAP', 'JAZZ_BRASS', 'JAZZ_CUBANO', 'JAZZ_DOUBLE_BASS', 'JAZZ_DRUMS', 'JAZZ_ELECTRIC_BASS', 'JAZZ_FUNK', 'JAZZ_FUSION', 'JAZZ_GUITAR', 'JAZZ_MEXICANO', 'JAZZ_PIANO', 'JAZZ_QUARTET', 'JAZZ_RAP', 'JAZZ_SAXOPHONE', 'JAZZ_TRIO', 'JAZZ_TRUMPET', 'JAZZ_VIOLIN', 'JDANCE', 'JPOP', 'JRAP', 'JUMP_UP', 'KC_INDIE', 'KENTUCKY_INDIE', 'KENT_INDIE', 'KHOP', 'KINDIE', 'KINGSTON_ON_INDIE', 'KOREAN_POP', 'KOREAN_RB', 'KOSOVAN_POP', 'KPOP', 'KPOP_BOY_GROUP', 'KPOP_GIRL_GROUP', 'KWAITO_HOUSE', 'LATIN', 'LATINTRONICA', 'LATIN_ALTERNATIVE', 'LATIN_ARENA_POP', 'LATIN_HIP_HOP', 'LATIN_JAZZ', 'LATIN_POP', 'LATIN_ROCK', 'LATIN_TALENT_SHOW', 'LATIN_TECH_HOUSE', 'LATIN_VIRAL_POP', 'LA_INDIE', 'LA_POP', 'LEEDS_INDIE', 'LEICESTER_INDIE', 'LEIPZIG_ELECTRONIC', 'LGBTQ_HIP_HOP', 'LILITH', 'LIQUID_FUNK', 'LITHUANIAN_ELECTRONIC', 'LIVERPOOL_INDIE', 'LOFI_BEATS', 'LOFI_HOUSE', 'LONDON_INDIE', 'LONDON_ON_INDIE', 'LONDON_RAP', 'LOUISVILLE_INDIE', 'LOUNGE', 'MALAYSIAN_POP', 'MANCHESTER_HIP_HOP', 'MANCHESTER_INDIE', 'MANDIBLE', 'MANITOBA_INDIE', 'MASHUP', 'MELBOURNE_INDIE', 'MELLOW_GOLD', 'MELODIC_HARDCORE', 'MELODIC_METALCORE', 'MELODIC_RAP', 'MELODIPOP', 'MEME_RAP', 'MEMPHIS_HIP_HOP', 'MEMPHIS_INDIE', 'MERENGUE', 'METAL', 'METALCORE', 'METROPOPOLIS', 'MEXICAN_INDIE', 'MEXICAN_POP', 'MIAMI_HIP_HOP', 'MIAMI_INDIE', 'MICHIGAN_INDIE', 'MICROHOUSE', 'MILAN_INDIE', 'MILWAUKEE_INDIE', 'MINIMAL_DUB', 'MINIMAL_DUBSTEP', 'MINIMAL_TECHNO', 'MINIMAL_TECH_HOUSE', 'MINNEAPOLIS_INDIE', 'MINNEAPOLIS_SOUND', 'MINNESOTA_HIP_HOP', 'MODERN_ALTERNATIVE_ROCK', 'MODERN_BLUES', 'MODERN_BLUES_ROCK', 'MODERN_BOLLYWOOD', 'MODERN_COUNTRY_ROCK', 'MODERN_HARD_ROCK', 'MODERN_REGGAE', 'MODERN_ROCK', 'MODERN_SALSA', 'MONTREAL_INDIE', 'MOOMBAHTON', 'MOROCCAN_POP', 'MOVIE_TUNES', 'MPB', 'MUNICH_ELECTRONIC', 'MUNICH_INDIE', 'MUSICA_CANARIA', 'MUSIQUE_CONCRETE', 'NASHVILLE_INDIE', 'NASHVILLE_SINGERSONGWRITER', 'NASHVILLE_SOUND', 'NATIVE_AMERICAN_HIP_HOP', 'NC_HIP_HOP', 'NEOCLASSICAL', 'NEON_POP_PUNK', 'NEOPSYCHEDELIC', 'NEOROCKABILLY', 'NEOSINGERSONGWRITER', 'NEOTRADITIONAL_COUNTRY', 'NEO_MELLOW', 'NEO_RB', 'NEO_SOUL', 'NEUROFUNK', 'NEWCASTLE_INDIE', 'NEWCASTLE_NSW_INDIE', 'NEWFOUNDLAND_INDIE', 'NEW_AMERICANA', 'NEW_FRENCH_TOUCH', 'NEW_ISOLATIONISM', 'NEW_JACK_SWING', 'NEW_JERSEY_INDIE', 'NEW_JERSEY_RAP', 'NEW_ORLEANS_FUNK', 'NEW_ORLEANS_JAZZ', 'NEW_ORLEANS_RAP', 'NEW_RAVE', 'NEW_WAVE_POP', 'NIGERIAN_HIP_HOP', 'NIGERIAN_POP', 'NINJA', 'NOISE_POP', 'NORDIC_HOUSE', 'NORTENO', 'NORTH_EAST_ENGLAND_INDIE', 'NORWEGIAN_INDIE', 'NORWEGIAN_JAZZ', 'NORWEGIAN_POP', 'NORWEGIAN_TECHNO', 'NOTTINGHAM_INDIE', 'NOVA_MPB', 'NUMETALCORE', 'NU_AGE', 'NU_DISCO', 'NU_GAZE', 'NU_JAZZ', 'NU_METAL', 'NYC_POP', 'NYC_RAP', 'NZ_ELECTRONIC', 'NZ_HIP_HOP', 'NZ_POP', 'OAKLAND_INDIE', 'OKC_INDIE', 'OLYMPIA_WA_INDIE', 'ONTARIO_INDIE', 'ORGANIC_ELECTRONIC', 'ORLANDO_INDIE', 'OSLO_INDIE', 'OTACORE', 'OTTAWA_INDIE', 'OUTLAW_COUNTRY', 'OUTSIDER_HOUSE', 'OXFORD_INDIE', 'PAGODE', 'PANAMANIAN_POP', 'PARTYSCHLAGER', 'PEI_INDIE', 'PERMANENT_WAVE', 'PERREO', 'PERTH_INDIE', 'PHILLY_INDIE', 'PHILLY_RAP', 'PHONK', 'PIANO_ROCK', 'PINOY_INDIE', 'PITTSBURGH_INDIE', 'PITTSBURGH_RAP', 'PIXIE', 'POLISH_ELECTRONICA', 'POLISH_JAZZ', 'POP', 'POPPING', 'POP_ARGENTINO', 'POP_CATRACHO', 'POP_EDM', 'POP_EMO', 'POP_FOLK', 'POP_HOUSE', 'POP_NACIONAL', 'POP_PUNK', 'POP_QUEBECOIS', 'POP_RAP', 'POP_REGGAETON', 'POP_ROCK', 'POP_URBAINE', 'PORTLAND_HIP_HOP', 'PORTLAND_INDIE', 'PORTSMOUTH_INDIE', 'PORTUGUESE_CONTEMPORARY_CLASSICAL', 'PORTUGUESE_JAZZ', 'POSTGRUNGE', 'POSTHARDCORE', 'POSTSCREAMO', 'POSTTEEN_POP', 'PROGRESSIVE_ELECTRO_HOUSE', 'PROGRESSIVE_HOUSE', 'PROGRESSIVE_JAZZ_FUSION', 'PROGRESSIVE_METAL', 'PROGRESSIVE_POSTHARDCORE', 'PROGRESSIVE_TRANCE', 'PROGRESSIVE_TRANCE_HOUSE', 'PSYCHEDELIC_ROCK', 'PUERTO_RICAN_INDIE', 'PUERTO_RICAN_POP', 'PUNK', 'QUEBEC_INDIE', 'RAP', 'RAP_CATALAN', 'RAP_CONSCIENT', 'RAP_DOMINICANO', 'RAP_KREYOL', 'RAP_LATINA', 'RAP_MARSEILLE', 'RAP_METAL', 'RAWSTYLE', 'RAW_TECHNO', 'RB', 'RB_BRASILEIRO', 'RB_EN_ESPANOL', 'REBEL_BLUES', 'REGGAETON', 'REGGAETON_CHILENO', 'REGGAETON_FLOW', 'REGGAE_EN_ESPANOL', 'REGGAE_FUSION', 'REGGAE_ROCK', 'REGIONAL_MEXICAN', 'REGIONAL_MEXICAN_POP', 'RETRO_SOUL', 'ROCHESTER_MN_INDIE', 'ROCK', 'ROCKABILLY', 'ROCK_EN_ESPANOL', 'ROMANIAN_POP', 'ROOTS_AMERICANA', 'RVA_INDIE', 'SAN_DIEGO_INDIE', 'SAN_DIEGO_RAP', 'SCANDINAVIAN_RB', 'SCANDIPOP', 'SCOTTISH_ELECTRONIC', 'SCOTTISH_HIP_HOP', 'SCOTTISH_INDIE', 'SCOTTISH_INDIE_ROCK', 'SCOTTISH_ROCK', 'SCREAMO', 'SCREAM_RAP', 'SEATTLE_INDIE', 'SERTANEJO', 'SERTANEJO_POP', 'SERTANEJO_UNIVERSITARIO', 'SHIMMER_POP', 'SHIMMER_PSYCH', 'SHIVER_POP', 'SHOEGAZE', 'SHOW_TUNES', 'SINGAPOREAN_POP', 'SKATE_PUNK', 'SKWEEE', 'SKY_ROOM', 'SLAYER', 'SMALL_ROOM', 'SMOOTH_JAZZ', 'SMOOTH_SAXOPHONE', 'SOCAL_POP_PUNK', 'SOCIAL_MEDIA_POP', 'SOFT_ROCK', 'SOUL', 'SOUND_ART', 'SOUTHAMPTON_INDIE', 'SOUTHERN_HIP_HOP', 'SOUTH_AFRICAN_ALTERNATIVE', 'SOUTH_AFRICAN_HIP_HOP', 'SOUTH_AFRICAN_POP', 'SOUTH_AFRICAN_ROCK', 'SPANISH_HIP_HOP', 'SPANISH_INDIE_POP', 'SPANISH_NOISE_POP', 'SPANISH_POP', 'SPEED_GARAGE', 'STOCKHOLM_INDIE', 'STOMP_AND_HOLLER', 'SUBSTEP', 'SWEDISH_ALTERNATIVE_ROCK', 'SWEDISH_ELECTRONIC', 'SWEDISH_ELECTROPOP', 'SWEDISH_GANGSTA_RAP', 'SWEDISH_HIP_HOP', 'SWEDISH_IDOL_POP', 'SWEDISH_INDIE_ROCK', 'SWEDISH_JAZZ', 'SWEDISH_POP', 'SWEDISH_REGGAE', 'SWEDISH_SINGERSONGWRITER', 'SWEDISH_SOUL', 'SWEDISH_TECHNO', 'SWEDISH_TROPICAL_HOUSE', 'SWEDISH_URBAN', 'SWING', 'SWISS_INDIE', 'SYDNEY_INDIE', 'TALENT_SHOW', 'TECHNO', 'TECH_HOUSE', 'TEEN_POP', 'TEXAS_COUNTRY', 'TIJUANA_ELECTRONIC', 'TORONTO_INDIE', 'TORONTO_RAP', 'TRANCECORE', 'TRANSPOP', 'TRAP', 'TRAPRUN', 'TRAP_ARGENTINO', 'TRAP_BRASILEIRO', 'TRAP_CATALA', 'TRAP_CHILENO', 'TRAP_ESPANOL', 'TRAP_FRANCAIS', 'TRAP_LATINO', 'TRAP_QUEEN', 'TRAP_SOUL', 'TRIBAL_HOUSE', 'TROPICAL', 'TROPICAL_HOUSE', 'TURKISH_HIP_HOP', 'TURKISH_JAZZ', 'TURKISH_POP', 'UAE_INDIE', 'UKRAINIAN_POP', 'UK_ALTERNATIVE_HIP_HOP', 'UK_ALTERNATIVE_POP', 'UK_AMERICANA', 'UK_CONTEMPORARY_RB', 'UK_DANCE', 'UK_DANCEHALL', 'UK_DNB', 'UK_DRILL', 'UK_EXPERIMENTAL_ELECTRONIC', 'UK_FUNKY', 'UK_HIP_HOP', 'UK_HOUSE', 'UK_POP', 'UK_TECH_HOUSE', 'UNDERGROUND_HIP_HOP', 'UPLIFTING_TRANCE', 'URBAN_CONTEMPORARY', 'UTAH_INDIE', 'VANCOUVER_INDIE', 'VAPOR_HOUSE', 'VAPOR_POP', 'VAPOR_SOUL', 'VAPOR_TRAP', 'VAPOR_TWITCH', 'VENEZUELAN_HIP_HOP', 'VERACRUZ_INDIE', 'VERMONT_INDIE', 'VICTORIA_BC_INDIE', 'VIRAL_POP', 'VIRAL_TRAP', 'VOCAL_HOUSE', 'VOCAL_JAZZ', 'WARM_DRONE', 'WAVE', 'WELSH_INDIE', 'WEST_AUSTRALIAN_HIP_HOP', 'WEST_COAST_TRAP', 'WISCONSIN_INDIE', 'WITCH_HOUSE', 'WONKY', 'WORLD_FUSION', 'ZAPSTEP', 'ZIMDANCEHALL']

def read_file(tsv_file):
    tracks = {}
    with open(tsv_file) as fp:
        reader = csv.reader(fp, delimiter='\t')
        next(reader, None)  # skip header
        for row in reader:
            track_id = row[0]
            tracks[track_id] = {
                'path': row[3].replace('.mp3', '.npy').replace(".m4a", '.npy'),
                'tags': row[5:],
            }
    return tracks



class Solver(object):
    def __init__(self, data_loader, config):
        # data loader
        self.data_loader = data_loader
        self.dataset = config.dataset
        self.data_path = config.data_path
        self.input_length = config.input_length

        # training settings
        self.n_epochs = config.n_epochs
        self.lr = config.lr
        self.use_tensorboard = config.use_tensorboard

        # model path and step size
        self.model_save_path = config.model_save_path
        self.model_load_path = config.model_load_path
        self.log_step = config.log_step
        self.batch_size = config.batch_size
        self.model_type = config.model_type

        # cuda
        self.is_cuda = torch.cuda.is_available()

        # Build model
        self.get_dataset()
        self.build_model()

        # Tensorboard
        self.writer = SummaryWriter()

    def get_dataset(self):
        if self.dataset == 'mtat':
            self.valid_list = np.load('./../split/mtat/valid.npy')
            self.binary = np.load('./../split/mtat/binary.npy')
        if self.dataset == 'msd':
            train_file = os.path.join('./../split/msd','filtered_list_train.cP')
            train_list = pickle.load(open(train_file,'rb'), encoding='bytes')
            val_set = train_list[201680:]
            self.valid_list = [value for value in val_set if value.decode() not in skip_files]
            id2tag_file = os.path.join('./../split/msd', 'msd_id_to_tag_vector.cP')
            self.id2tag = pickle.load(open(id2tag_file,'rb'), encoding='bytes')
        if self.dataset == 'jamendo':
            train_file = os.path.join('./../split/mtg-jamendo', 'autotagging_top50tags-validation.tsv')
            self.file_dict= read_file(train_file)
            self.valid_list= list(read_file(train_file).keys())
            self.mlb = LabelBinarizer().fit(TAGS)
        if self.dataset == 'jamendo-mood':
            train_file = os.path.join('./../split/mtg-jamendo-mood', 'autotagging_moodtheme-validation.tsv') # why validation instead of train?
            self.file_dict= read_file(train_file)
            self.valid_list= list(read_file(train_file).keys())
            self.mlb = LabelBinarizer().fit(TAGS_MOOD)
        if self.dataset == 'genres':
            train_file = os.path.join('./../split/genres', 'validation.tsv') # why validation instead of train?
            self.file_dict= read_file(train_file)
            self.valid_list= list(read_file(train_file).keys())
            self.mlb = LabelBinarizer().fit(genres_tags)


    def get_model(self):
        if self.model_type == 'fcn':
            return Model.FCN()
        elif self.model_type == 'musicnn':
            return Model.Musicnn(dataset=self.dataset)
        elif self.model_type == 'crnn':
            return Model.CRNN()
        elif self.model_type == 'sample':
            return Model.SampleCNN()
        elif self.model_type == 'se':
            return Model.SampleCNNSE()
        elif self.model_type == 'short':
            return Model.ShortChunkCNN()
        elif self.model_type == 'short_res':
            return Model.ShortChunkCNN_Res()
        elif self.model_type == 'attention':
            return Model.CNNSA()
        elif self.model_type == 'hcnn':
            return Model.HarmonicCNN()

    def build_model(self):
        # model
        self.model = self.get_model()

        # cuda
        if self.is_cuda:
            self.model.cuda()

        # load pretrained model
        if len(self.model_load_path) > 1:
            self.load(self.model_load_path)

        # optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, weight_decay=1e-4)

    def load(self, filename):
        S = torch.load(filename)
        if 'spec.mel_scale.fb' in S.keys():
            self.model.spec.mel_scale.fb = S['spec.mel_scale.fb']
        self.model.load_state_dict(S)

    def to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def get_loss_function(self):
        return nn.BCELoss()

    def train(self):
        # Start training
        start_t = time.time()
        current_optimizer = 'adam'
        reconst_loss = self.get_loss_function()
        best_metric = 0
        drop_counter = 0

        # Iterate
        for epoch in range(self.n_epochs):
            ctr = 0
            drop_counter += 1
            self.model = self.model.train()
            for x, y in self.data_loader:
                ctr += 1
                # Forward
                x = self.to_var(x)
                y = self.to_var(y)
                out = self.model(x)

                # Backward
                #print("SHAPES")
                #print(out.shape)
                #print(y.shape)
                loss = reconst_loss(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Log
                self.print_log(epoch, ctr, loss, start_t)
            self.writer.add_scalar('Loss/train', loss.item(), epoch)

            # validation
            best_metric = self.validation(best_metric, epoch)

            # schedule optimizer
            current_optimizer, drop_counter = self.opt_schedule(current_optimizer, drop_counter)

        print("[%s] Train finished. Elapsed: %s"
                % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    datetime.timedelta(seconds=time.time() - start_t)))

    def opt_schedule(self, current_optimizer, drop_counter):
        # adam to sgd
        if current_optimizer == 'adam' and drop_counter == 80:
            self.load(os.path.join(self.model_save_path, 'best_model.pth'))
            self.optimizer = torch.optim.SGD(self.model.parameters(), 0.001,
                                            momentum=0.9, weight_decay=0.0001,
                                            nesterov=True)
            current_optimizer = 'sgd_1'
            drop_counter = 0
            print('sgd 1e-3')
        # first drop
        if current_optimizer == 'sgd_1' and drop_counter == 20:
            self.load(os.path.join(self.model_save_path, 'best_model.pth'))
            for pg in self.optimizer.param_groups:
                pg['lr'] = 0.0001
            current_optimizer = 'sgd_2'
            drop_counter = 0
            print('sgd 1e-4')
        # second drop
        if current_optimizer == 'sgd_2' and drop_counter == 20:
            self.load(os.path.join(self.model_save_path, 'best_model.pth'))
            for pg in self.optimizer.param_groups:
                pg['lr'] = 0.00001
            current_optimizer = 'sgd_3'
            print('sgd 1e-5')
        return current_optimizer, drop_counter

    def save(self, filename):
        model = self.model.state_dict()
        torch.save({'model': model}, filename)

    def get_tensor(self, fn):
        # load audio
        if self.dataset == 'mtat':
            npy_path = os.path.join(self.data_path, 'mtat', 'npy', fn.split('/')[1][:-3]) + 'npy'
        elif self.dataset == 'msd':
            msid = fn.decode()
            filename = '{}/{}/{}/{}.npy'.format(msid[2], msid[3], msid[4], msid)
            npy_path = os.path.join(self.data_path, filename)
        elif self.dataset == 'jamendo':
            filename = self.file_dict[fn]['path']
            npy_path = os.path.join(self.data_path, filename)
        elif self.dataset in ['jamendo-mood', 'genres']:
            filename = self.file_dict[fn]['path']
            npy_path = os.path.join(self.data_path, filename.split("/")[-1])
        raw = np.load(npy_path, mmap_mode='r')

        # split chunk
        length = len(raw)
        hop = (length - self.input_length) // self.batch_size
        x = torch.zeros(self.batch_size, self.input_length)
        for i in range(self.batch_size):
            x[i] = torch.Tensor(raw[i*hop:i*hop+self.input_length]).unsqueeze(0)
        return x

    def get_auc(self, est_array, gt_array):
        try:
            roc_aucs  = metrics.roc_auc_score(gt_array, est_array, average='macro')
        except:
            roc_aucs = 1.
        try:
            pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
        except:
            pr_aucs = 1.
        print('roc_auc: %.4f' % roc_aucs)
        print('pr_auc: %.4f' % pr_aucs)
        return roc_aucs, pr_aucs

    def print_log(self, epoch, ctr, loss, start_t):
        if (ctr) % self.log_step == 0:
            print("[%s] Epoch [%d/%d] Iter [%d/%d] train loss: %.4f Elapsed: %s" %
                    (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        epoch+1, self.n_epochs, ctr, len(self.data_loader), loss.item(),
                        datetime.timedelta(seconds=time.time()-start_t)))

    def validation(self, best_metric, epoch):
        roc_auc, pr_auc, loss = self.get_validation_score(epoch)
        score = 1 - loss
        if score > best_metric:
            print('best model!')
            best_metric = score
            torch.save(self.model.state_dict(),
                       os.path.join(self.model_save_path, 'best_model.pth'))
        return best_metric


    def get_validation_score(self, epoch):
        self.model = self.model.eval()
        est_array = []
        gt_array = []
        losses = []
        reconst_loss = self.get_loss_function()
        index = 0
        for line in tqdm.tqdm(self.valid_list):
            if self.dataset == 'mtat':
                ix, fn = line.split('\t')
            elif self.dataset == 'msd':
                fn = line
                if fn.decode() in skip_files:
                    continue
            elif self.dataset in ['jamendo', 'jamendo-mood','genres']:
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
            index += 1

        est_array, gt_array = np.array(est_array), np.array(gt_array)
        loss = np.mean(losses)
        print('loss: %.4f' % loss)

        roc_auc, pr_auc = self.get_auc(est_array, gt_array)
        self.writer.add_scalar('Loss/valid', loss, epoch)
        self.writer.add_scalar('AUC/ROC', roc_auc, epoch)
        self.writer.add_scalar('AUC/PR', pr_auc, epoch)
        return roc_auc, pr_auc, loss

