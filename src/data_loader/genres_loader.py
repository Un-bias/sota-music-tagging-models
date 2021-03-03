# coding: utf-8
import pickle
import os
import csv
import numpy as np
from torch.utils import data
from sklearn.preprocessing import LabelBinarizer

META_PATH = '/home/jupyter/models/sota/split/genres'

TAGS = ['ABSTRACTRO', 'ABSTRACT_BEATS', 'ABSTRACT_HIP_HOP', 'ACID_HOUSE', 'ACID_JAZZ', 'ACID_TECHNO', 'ACOUSTIC_POP', 'ADELAIDE_INDIE', 'ADULT_STANDARDS', 'AFRICAN_ELECTRONIC', 'AFROBEAT', 'AFROPOP', 'AFRO_DANCEHALL', 'AFRO_HOUSE', 'ALABAMA_INDIE', 'ALABAMA_RAP', 'ALASKA_INDIE', 'ALBANIAN_HIP_HOP', 'ALBANIAN_POP', 'ALBERTA_COUNTRY', 'ALBERTA_HIP_HOP', 'ALBUM_ROCK', 'ALBUQUERQUE_INDIE', 'ALTERNATIVE_AMERICANA', 'ALTERNATIVE_COUNTRY', 'ALTERNATIVE_DANCE', 'ALTERNATIVE_EMO', 'ALTERNATIVE_HIP_HOP', 'ALTERNATIVE_METAL', 'ALTERNATIVE_POP', 'ALTERNATIVE_POP_ROCK', 'ALTERNATIVE_RB', 'ALTERNATIVE_ROCK', 'ALTERNATIVE_ROOTS_ROCK', 'AMBEAT', 'AMBIENT', 'AMBIENT_FOLK', 'AMBIENT_IDM', 'AMBIENT_TECHNO', 'AMERICAN_SHOEGAZE', 'ANN_ARBOR_INDIE', 'ANTHEM_EMO', 'ANTHEM_WORSHIP', 'ANTIFOLK', 'ARABIC_HIP_HOP', 'ARGENTINE_HIP_HOP', 'ARGENTINE_INDIE', 'ARGENTINE_ROCK', 'ARKANSAS_COUNTRY', 'ART_POP', 'ART_ROCK', 'ASBURY_PARK_INDIE', 'ASHEVILLE_INDIE', 'ATLANTA_INDIE', 'ATL_HIP_HOP', 'ATL_TRAP', 'AUCKLAND_INDIE', 'AUSSIETRONICA', 'AUSSIE_EMO', 'AUSTINDIE', 'AUSTRALIAN_ALTERNATIVE_POP', 'AUSTRALIAN_ALTERNATIVE_ROCK', 'AUSTRALIAN_COUNTRY', 'AUSTRALIAN_DANCE', 'AUSTRALIAN_ELECTROPOP', 'AUSTRALIAN_GARAGE_PUNK', 'AUSTRALIAN_HIP_HOP', 'AUSTRALIAN_HOUSE', 'AUSTRALIAN_INDIE', 'AUSTRALIAN_INDIE_ROCK', 'AUSTRALIAN_POP', 'AUSTRALIAN_PSYCH', 'AUSTRALIAN_RB', 'AUSTRALIAN_REGGAE_FUSION', 'AUSTRALIAN_SHOEGAZE', 'AUSTRALIAN_SINGERSONGWRITER', 'AUSTRALIAN_TALENT_SHOW', 'AUSTRALIAN_TRAP', 'AUSTRALIAN_UNDERGROUND_HIP_HOP', 'AVANTGARDE', 'AVANTGARDE_JAZZ', 'AZONTO', 'AZONTOBEATS', 'A_CAPPELLA', 'BACHATA', 'BAILE_POP', 'BALEARIC', 'BALTIMORE_HIP_HOP', 'BALTIMORE_INDIE', 'BANDA', 'BANJO', 'BARBADIAN_POP', 'BAROQUE_POP', 'BASSHALL', 'BASSLINE', 'BASS_HOUSE', 'BASS_MUSIC', 'BASS_TRAP', 'BASS_TRIP', 'BATIDA', 'BATON_ROUGE_RAP', 'BATTLE_RAP', 'BAY_AREA_INDIE', 'BBOY', 'BC_UNDERGROUND_HIP_HOP', 'BEBOP', 'BEDROOM_POP', 'BEDROOM_SOUL', 'BELFAST_INDIE', 'BELGIAN_DANCE', 'BELGIAN_EDM', 'BELGIAN_HIP_HOP', 'BELGIAN_INDIE', 'BELGIAN_INDIE_ROCK', 'BELGIAN_MODERN_JAZZ', 'BELGIAN_POP', 'BELGIAN_ROCK', 'BERGEN_INDIE', 'BIG_BAND', 'BIG_BEAT', 'BIG_ROOM', 'BIRMINGHAM_GRIME', 'BIRMINGHAM_HIP_HOP', 'BIRMINGHAM_INDIE', 'BIRMINGHAM_METAL', 'BLUES_ROCK', 'BMORE', 'BOOGALOO', 'BOOM_BAP', 'BOSSA_NOVA', 'BOSSA_NOVA_COVER', 'BOSTON_HIP_HOP', 'BOSTON_INDIE', 'BOSTON_ROCK', 'BOUNCE', 'BOW_POP', 'BOY_BAND', 'BRASS_BAND', 'BRAZILIAN_EDM', 'BRAZILIAN_HIP_HOP', 'BRAZILIAN_HOUSE', 'BRAZILIAN_MODERN_JAZZ', 'BRAZILIAN_PSYCHEDELIC', 'BRAZILIAN_SOUL', 'BREAKBEAT', 'BREGA_FUNK', 'BRIGHTON_INDIE', 'BRILL_BUILDING_POP', 'BRISBANE_INDIE', 'BRISTOL_INDIE', 'BRITISH_ALTERNATIVE_ROCK', 'BRITISH_EXPERIMENTAL', 'BRITISH_FOLK', 'BRITISH_INDIE_ROCK', 'BRITISH_INVASION', 'BRITISH_JAZZ', 'BRITISH_SINGERSONGWRITER', 'BRITISH_SOUL', 'BRITPOP', 'BROADWAY', 'BROKEN_BEAT', 'BROOKLYN_INDIE', 'BROSTEP', 'BUBBLEGUM_DANCE', 'BULGARIAN_INDIE', 'CALI_RAP', 'CALMING_INSTRUMENTAL', 'CAMBRIDGESHIRE_INDIE', 'CANADIAN_CONTEMPORARY_COUNTRY', 'CANADIAN_CONTEMPORARY_RB', 'CANADIAN_COUNTRY', 'CANADIAN_ELECTRONIC', 'CANADIAN_ELECTROPOP', 'CANADIAN_EXPERIMENTAL', 'CANADIAN_FOLK', 'CANADIAN_HIP_HOP', 'CANADIAN_INDIE', 'CANADIAN_LATIN', 'CANADIAN_MODERN_JAZZ', 'CANADIAN_POP', 'CANADIAN_POP_PUNK', 'CANADIAN_POSTHARDCORE', 'CANADIAN_PUNK', 'CANADIAN_ROCK', 'CANADIAN_SHOEGAZE', 'CANADIAN_SINGERSONGWRITER', 'CANDY_POP', 'CANTAUTOR', 'CAPE_TOWN_INDIE', 'CATSTEP', 'CCM', 'CDMX_INDIE', 'CEDM', 'CELTIC_ROCK', 'CHAMBER_POP', 'CHAMBER_PSYCH', 'CHAMPETA', 'CHANNEL_ISLANDS_INDIE', 'CHANNEL_POP', 'CHARLOTTESVILLE_INDIE', 'CHARLOTTE_NC_INDIE', 'CHICAGO_DRILL', 'CHICAGO_HOUSE', 'CHICAGO_INDIE', 'CHICAGO_PUNK', 'CHICAGO_RAP', 'CHILEAN_INDIE', 'CHILLHOP', 'CHILLSTEP', 'CHILLWAVE', 'CHINESE_HIP_HOP', 'CHINESE_INDIE', 'CHRISTCHURCH_INDIE', 'CHRISTIAN_ALTERNATIVE_ROCK', 'CHRISTIAN_HIP_HOP', 'CHRISTIAN_INDIE', 'CHRISTIAN_MUSIC', 'CHRISTIAN_POP', 'CHRISTIAN_TRAP', 'CHRISTLICHER_RAP', 'CIRCUIT', 'CLASSIC_COUNTRY_POP', 'CLASSIC_ITALIAN_POP', 'CLASSIC_ROCK', 'CLASSIC_SWEDISH_POP', 'CLASSIFY', 'COLLAGE_POP', 'COLOGNE_INDIE', 'COLOMBIAN_HIP_HOP', 'COLOMBIAN_INDIE', 'COLOMBIAN_POP', 'COLUMBUS_OHIO_INDIE', 'COMPLEXTRO', 'COMPOSITIONAL_AMBIENT', 'CONNECTICUT_INDIE', 'CONSCIOUS_HIP_HOP', 'CONTEMPORARY_COUNTRY', 'CONTEMPORARY_JAZZ', 'CONTEMPORARY_POSTBOP', 'COOL_JAZZ', 'COUNTRY', 'COUNTRY_DAWN', 'COUNTRY_POP', 'COUNTRY_RAP', 'COUNTRY_ROAD', 'COUNTRY_ROCK', 'COVERCHILL', 'CRUNK', 'CUBAN_RUMBA', 'CUMBIA', 'CUMBIA_POP', 'DANCEHALL', 'DANCEPUNK', 'DANCE_POP', 'DANCE_ROCK', 'DANISH_ALTERNATIVE_ROCK', 'DANISH_ELECTRONIC', 'DANISH_ELECTROPOP', 'DANISH_INDIE_POP', 'DANISH_JAZZ', 'DANISH_METAL', 'DANISH_POP', 'DARK_DISCO', 'DARK_JAZZ', 'DARK_POSTPUNK', 'DARK_TECHNO', 'DARK_TRAP', 'DC_INDIE', 'DEEP_BIG_ROOM', 'DEEP_DISCO_HOUSE', 'DEEP_DUBSTEP', 'DEEP_EURO_HOUSE', 'DEEP_GERMAN_HIP_HOP', 'DEEP_GROOVE_HOUSE', 'DEEP_HOUSE', 'DEEP_IDM', 'DEEP_LATIN_ALTERNATIVE', 'DEEP_LIQUID_BASS', 'DEEP_MELODIC_EURO_HOUSE', 'DEEP_MINIMAL_TECHNO', 'DEEP_NEW_AMERICANA', 'DEEP_POP_EDM', 'DEEP_POP_RB', 'DEEP_SOUL_HOUSE', 'DEEP_SOUTHERN_TRAP', 'DEEP_TALENT_SHOW', 'DEEP_TECHNO', 'DEEP_TECH_HOUSE', 'DEEP_TROPICAL_HOUSE', 'DEEP_UNDERGROUND_HIP_HOP', 'DEMBOW', 'DENTON_TX_INDIE', 'DENVER_INDIE', 'DERRY_INDIE', 'DESI_HIP_HOP', 'DESI_POP', 'DESTROY_TECHNO', 'DETROIT_HIP_HOP', 'DETROIT_INDIE', 'DETROIT_TECHNO', 'DETROIT_TRAP', 'DEVON_INDIE', 'DFW_RAP', 'DIRTY_SOUTH_RAP', 'DISCO', 'DISCO_HOUSE', 'DIVA_HOUSE', 'DIY_EMO', 'DMV_RAP', 'DOMINICAN_POP', 'DOWNTEMPO', 'DREAMGAZE', 'DREAMO', 'DREAM_POP', 'DRIFT', 'DRILL', 'DRILL_AND_BASS', 'DRONE', 'DRUMFUNK', 'DRUM_AND_BASS', 'DUBLIN_INDIE', 'DUBSTEP', 'DUB_TECHNO', 'DUTCH_CABARET', 'DUTCH_EXPERIMENTAL_ELECTRONIC', 'DUTCH_HIP_HOP', 'DUTCH_HOUSE', 'DUTCH_INDIE', 'DUTCH_JAZZ', 'DUTCH_POP', 'DUTCH_ROCK', 'DUTCH_URBAN', 'EAST_ANGLIA_INDIE', 'EAST_COAST_HIP_HOP', 'EASYCORE', 'EASY_LISTENING', 'EAU_CLAIRE_INDIE', 'ECTOFOLK', 'EDINBURGH_INDIE', 'EDM', 'EDMONTON_INDIE', 'ELECTRA', 'ELECTRIC_BLUES', 'ELECTROCLASH', 'ELECTROFOX', 'ELECTRONICA', 'ELECTRONIC_ROCK', 'ELECTRONIC_TRAP', 'ELECTROPOP', 'ELECTROPOWERPOP', 'ELECTRO_HOUSE', 'ELECTRO_LATINO', 'ELECTRO_SWING', 'EL_PASO_INDIE', 'EMO', 'EMO_RAP', 'ENGLISH_INDIE_ROCK', 'ESCAPE_ROOM', 'ETHERPOP', 'ETHIOJAZZ', 'ETHNOTRONICA', 'EUPHORIC_HARDSTYLE', 'EURODANCE', 'EUROPOP', 'EUROVISION', 'EXPERIMENTAL', 'EXPERIMENTAL_AMBIENT', 'EXPERIMENTAL_ELECTRONIC', 'EXPERIMENTAL_FOLK', 'EXPERIMENTAL_HIP_HOP', 'EXPERIMENTAL_HOUSE', 'EXPERIMENTAL_POP', 'EXPERIMENTAL_PSYCH', 'EXPERIMENTAL_TECHNO', 'FIDGET_HOUSE', 'FILMI', 'FILTER_HOUSE', 'FILTHSTEP', 'FINNISH_EDM', 'FINNISH_ELECTRO', 'FINNISH_INDIE', 'FINNISH_JAZZ', 'FLOAT_HOUSE', 'FLORIDA_RAP', 'FLUXWORK', 'FOCUS', 'FOCUS_TRANCE', 'FOLK', 'FOLKPOP', 'FOLKTRONICA', 'FOLK_BRASILEIRO', 'FOLK_PUNK', 'FOLK_ROCK', 'FOOTWORK', 'FORRO', 'FORT_WORTH_INDIE', 'FOURTH_WORLD', 'FRANCOTON', 'FRANKFURT_ELECTRONIC', 'FREAK_FOLK', 'FREE_IMPROVISATION', 'FREE_JAZZ', 'FRENCH_HIP_HOP', 'FRENCH_INDIETRONICA', 'FRENCH_INDIE_POP', 'FRENCH_JAZZ', 'FRENCH_SHOEGAZE', 'FRENCH_TECHNO', 'FUNK', 'FUNKY_TECH_HOUSE', 'FUNK_CARIOCA', 'FUNK_DAS_ANTIGAS', 'FUNK_METAL', 'FUNK_OSTENTACAO', 'FUNK_ROCK', 'FUTURE_FUNK', 'FUTURE_GARAGE', 'FUTURE_HOUSE', 'GANGSTER_RAP', 'GARAGE_POP', 'GARAGE_PSYCH', 'GARAGE_PUNK', 'GARAGE_ROCK', 'GAUZE_POP', 'GERMAN_CLOUD_RAP', 'GERMAN_DANCE', 'GERMAN_HIP_HOP', 'GERMAN_HOUSE', 'GERMAN_INDIE', 'GERMAN_INDIE_FOLK', 'GERMAN_INDIE_ROCK', 'GERMAN_JAZZ', 'GERMAN_METAL', 'GERMAN_POP', 'GERMAN_ROCK', 'GERMAN_TECHNO', 'GHANAIAN_HIP_HOP', 'GIRL_GROUP', 'GLAM_METAL', 'GLAM_ROCK', 'GLASGOW_INDIE', 'GLITCH', 'GOSPEL', 'GOSPEL_RB', 'GOTHENBURG_INDIE', 'GQOM', 'GRAND_RAPIDS_INDIE', 'GRAVE_WAVE', 'GREEK_HOUSE', 'GRIME', 'GRIMEWAVE', 'GROOVE_METAL', 'GROOVE_ROOM', 'GRUNGE', 'G_FUNK', 'HALIFAX_INDIE', 'HAMBURG_ELECTRONIC', 'HAMBURG_HIP_HOP', 'HARDCORE_HIP_HOP', 'HARDCORE_TECHNO', 'HARDSTYLE', 'HARD_ROCK', 'HAWAIIAN_HIP_HOP', 'HEARTLAND_ROCK', 'HINRG', 'HIP_HOP', 'HIP_HOP_QUEBECOIS', 'HIP_HOUSE', 'HIP_POP', 'HOLLYWOOD', 'HOPEBEAT', 'HORROR_SYNTH', 'HOUSE', 'HOUSTON_INDIE', 'HOUSTON_RAP', 'HYPERPOP', 'HYPHY', 'ICELANDIC_ELECTRONIC', 'ICELANDIC_INDIE', 'ICELANDIC_POP', 'ICELANDIC_ROCK', 'IDOL', 'INDIANA_INDIE', 'INDIECOUSTICA', 'INDIETRONICA', 'INDIE_ANTHEMFOLK', 'INDIE_CAFE_POP', 'INDIE_DEUTSCHRAP', 'INDIE_DREAM_POP', 'INDIE_ELECTRONICA', 'INDIE_ELECTROPOP', 'INDIE_FOLK', 'INDIE_GARAGE_ROCK', 'INDIE_JAZZ', 'INDIE_POP', 'INDIE_POPTIMISM', 'INDIE_POP_RAP', 'INDIE_PSYCHPOP', 'INDIE_PUNK', 'INDIE_QUEBECOIS', 'INDIE_RB', 'INDIE_ROCK', 'INDIE_ROCKISM', 'INDIE_SHOEGAZE', 'INDIE_SOUL', 'INDIE_SURF', 'INDIE_TICO', 'INDONESIAN_EDM', 'INDONESIAN_HIP_HOP', 'INDONESIAN_JAZZ', 'INDONESIAN_POP', 'INDONESIAN_RB', 'INDUSTRIAL', 'INDUSTRIAL_METAL', 'INDY_INDIE', 'INSTRUMENTAL_FUNK', 'INSTRUMENTAL_GRIME', 'INTELLIGENT_DANCE_MUSIC', 'IRANIAN_EXPERIMENTAL', 'IRISH_ELECTRONIC', 'IRISH_HIP_HOP', 'IRISH_INDIE', 'IRISH_INDIE_ROCK', 'IRISH_POP', 'IRISH_ROCK', 'IRISH_SINGERSONGWRITER', 'ISLE_OF_WIGHT_INDIE', 'ISRAELI_HIP_HOP', 'ISRAELI_JAZZ', 'ISRAELI_POP', 'ITALIAN_ALTERNATIVE', 'ITALIAN_ARENA_POP', 'ITALIAN_HIP_HOP', 'ITALIAN_JAZZ', 'ITALIAN_POP', 'ITALIAN_TECHNO', 'ITALIAN_TECH_HOUSE', 'ITALO_DANCE', 'JACKSONVILLE_INDIE', 'JAMBIENT', 'JAMTRONICA', 'JAPANESE_CITY_POP', 'JAPANESE_EXPERIMENTAL', 'JAPANESE_JAZZ', 'JAPANESE_RB', 'JAZZ', 'JAZZTRONICA', 'JAZZ_BOOM_BAP', 'JAZZ_BRASS', 'JAZZ_CUBANO', 'JAZZ_DOUBLE_BASS', 'JAZZ_DRUMS', 'JAZZ_ELECTRIC_BASS', 'JAZZ_FUNK', 'JAZZ_FUSION', 'JAZZ_GUITAR', 'JAZZ_MEXICANO', 'JAZZ_PIANO', 'JAZZ_QUARTET', 'JAZZ_RAP', 'JAZZ_SAXOPHONE', 'JAZZ_TRIO', 'JAZZ_TRUMPET', 'JAZZ_VIOLIN', 'JDANCE', 'JPOP', 'JRAP', 'JUMP_UP', 'KC_INDIE', 'KENTUCKY_INDIE', 'KENT_INDIE', 'KHOP', 'KINDIE', 'KINGSTON_ON_INDIE', 'KOREAN_POP', 'KOREAN_RB', 'KOSOVAN_POP', 'KPOP', 'KPOP_BOY_GROUP', 'KPOP_GIRL_GROUP', 'KWAITO_HOUSE', 'LATIN', 'LATINTRONICA', 'LATIN_ALTERNATIVE', 'LATIN_ARENA_POP', 'LATIN_HIP_HOP', 'LATIN_JAZZ', 'LATIN_POP', 'LATIN_ROCK', 'LATIN_TALENT_SHOW', 'LATIN_TECH_HOUSE', 'LATIN_VIRAL_POP', 'LA_INDIE', 'LA_POP', 'LEEDS_INDIE', 'LEICESTER_INDIE', 'LEIPZIG_ELECTRONIC', 'LGBTQ_HIP_HOP', 'LILITH', 'LIQUID_FUNK', 'LITHUANIAN_ELECTRONIC', 'LIVERPOOL_INDIE', 'LOFI_BEATS', 'LOFI_HOUSE', 'LONDON_INDIE', 'LONDON_ON_INDIE', 'LONDON_RAP', 'LOUISVILLE_INDIE', 'LOUNGE', 'MALAYSIAN_POP', 'MANCHESTER_HIP_HOP', 'MANCHESTER_INDIE', 'MANDIBLE', 'MANITOBA_INDIE', 'MASHUP', 'MELBOURNE_INDIE', 'MELLOW_GOLD', 'MELODIC_HARDCORE', 'MELODIC_METALCORE', 'MELODIC_RAP', 'MELODIPOP', 'MEME_RAP', 'MEMPHIS_HIP_HOP', 'MEMPHIS_INDIE', 'MERENGUE', 'METAL', 'METALCORE', 'METROPOPOLIS', 'MEXICAN_INDIE', 'MEXICAN_POP', 'MIAMI_HIP_HOP', 'MIAMI_INDIE', 'MICHIGAN_INDIE', 'MICROHOUSE', 'MILAN_INDIE', 'MILWAUKEE_INDIE', 'MINIMAL_DUB', 'MINIMAL_DUBSTEP', 'MINIMAL_TECHNO', 'MINIMAL_TECH_HOUSE', 'MINNEAPOLIS_INDIE', 'MINNEAPOLIS_SOUND', 'MINNESOTA_HIP_HOP', 'MODERN_ALTERNATIVE_ROCK', 'MODERN_BLUES', 'MODERN_BLUES_ROCK', 'MODERN_BOLLYWOOD', 'MODERN_COUNTRY_ROCK', 'MODERN_HARD_ROCK', 'MODERN_REGGAE', 'MODERN_ROCK', 'MODERN_SALSA', 'MONTREAL_INDIE', 'MOOMBAHTON', 'MOROCCAN_POP', 'MOVIE_TUNES', 'MPB', 'MUNICH_ELECTRONIC', 'MUNICH_INDIE', 'MUSICA_CANARIA', 'MUSIQUE_CONCRETE', 'NASHVILLE_INDIE', 'NASHVILLE_SINGERSONGWRITER', 'NASHVILLE_SOUND', 'NATIVE_AMERICAN_HIP_HOP', 'NC_HIP_HOP', 'NEOCLASSICAL', 'NEON_POP_PUNK', 'NEOPSYCHEDELIC', 'NEOROCKABILLY', 'NEOSINGERSONGWRITER', 'NEOTRADITIONAL_COUNTRY', 'NEO_MELLOW', 'NEO_RB', 'NEO_SOUL', 'NEUROFUNK', 'NEWCASTLE_INDIE', 'NEWCASTLE_NSW_INDIE', 'NEWFOUNDLAND_INDIE', 'NEW_AMERICANA', 'NEW_FRENCH_TOUCH', 'NEW_ISOLATIONISM', 'NEW_JACK_SWING', 'NEW_JERSEY_INDIE', 'NEW_JERSEY_RAP', 'NEW_ORLEANS_FUNK', 'NEW_ORLEANS_JAZZ', 'NEW_ORLEANS_RAP', 'NEW_RAVE', 'NEW_WAVE_POP', 'NIGERIAN_HIP_HOP', 'NIGERIAN_POP', 'NINJA', 'NOISE_POP', 'NORDIC_HOUSE', 'NORTENO', 'NORTH_EAST_ENGLAND_INDIE', 'NORWEGIAN_INDIE', 'NORWEGIAN_JAZZ', 'NORWEGIAN_POP', 'NORWEGIAN_TECHNO', 'NOTTINGHAM_INDIE', 'NOVA_MPB', 'NUMETALCORE', 'NU_AGE', 'NU_DISCO', 'NU_GAZE', 'NU_JAZZ', 'NU_METAL', 'NYC_POP', 'NYC_RAP', 'NZ_ELECTRONIC', 'NZ_HIP_HOP', 'NZ_POP', 'OAKLAND_INDIE', 'OKC_INDIE', 'OLYMPIA_WA_INDIE', 'ONTARIO_INDIE', 'ORGANIC_ELECTRONIC', 'ORLANDO_INDIE', 'OSLO_INDIE', 'OTACORE', 'OTTAWA_INDIE', 'OUTLAW_COUNTRY', 'OUTSIDER_HOUSE', 'OXFORD_INDIE', 'PAGODE', 'PANAMANIAN_POP', 'PARTYSCHLAGER', 'PEI_INDIE', 'PERMANENT_WAVE', 'PERREO', 'PERTH_INDIE', 'PHILLY_INDIE', 'PHILLY_RAP', 'PHONK', 'PIANO_ROCK', 'PINOY_INDIE', 'PITTSBURGH_INDIE', 'PITTSBURGH_RAP', 'PIXIE', 'POLISH_ELECTRONICA', 'POLISH_JAZZ', 'POP', 'POPPING', 'POP_ARGENTINO', 'POP_CATRACHO', 'POP_EDM', 'POP_EMO', 'POP_FOLK', 'POP_HOUSE', 'POP_NACIONAL', 'POP_PUNK', 'POP_QUEBECOIS', 'POP_RAP', 'POP_REGGAETON', 'POP_ROCK', 'POP_URBAINE', 'PORTLAND_HIP_HOP', 'PORTLAND_INDIE', 'PORTSMOUTH_INDIE', 'PORTUGUESE_CONTEMPORARY_CLASSICAL', 'PORTUGUESE_JAZZ', 'POSTGRUNGE', 'POSTHARDCORE', 'POSTSCREAMO', 'POSTTEEN_POP', 'PROGRESSIVE_ELECTRO_HOUSE', 'PROGRESSIVE_HOUSE', 'PROGRESSIVE_JAZZ_FUSION', 'PROGRESSIVE_METAL', 'PROGRESSIVE_POSTHARDCORE', 'PROGRESSIVE_TRANCE', 'PROGRESSIVE_TRANCE_HOUSE', 'PSYCHEDELIC_ROCK', 'PUERTO_RICAN_INDIE', 'PUERTO_RICAN_POP', 'PUNK', 'QUEBEC_INDIE', 'RAP', 'RAP_CATALAN', 'RAP_CONSCIENT', 'RAP_DOMINICANO', 'RAP_KREYOL', 'RAP_LATINA', 'RAP_MARSEILLE', 'RAP_METAL', 'RAWSTYLE', 'RAW_TECHNO', 'RB', 'RB_BRASILEIRO', 'RB_EN_ESPANOL', 'REBEL_BLUES', 'REGGAETON', 'REGGAETON_CHILENO', 'REGGAETON_FLOW', 'REGGAE_EN_ESPANOL', 'REGGAE_FUSION', 'REGGAE_ROCK', 'REGIONAL_MEXICAN', 'REGIONAL_MEXICAN_POP', 'RETRO_SOUL', 'ROCHESTER_MN_INDIE', 'ROCK', 'ROCKABILLY', 'ROCK_EN_ESPANOL', 'ROMANIAN_POP', 'ROOTS_AMERICANA', 'RVA_INDIE', 'SAN_DIEGO_INDIE', 'SAN_DIEGO_RAP', 'SCANDINAVIAN_RB', 'SCANDIPOP', 'SCOTTISH_ELECTRONIC', 'SCOTTISH_HIP_HOP', 'SCOTTISH_INDIE', 'SCOTTISH_INDIE_ROCK', 'SCOTTISH_ROCK', 'SCREAMO', 'SCREAM_RAP', 'SEATTLE_INDIE', 'SERTANEJO', 'SERTANEJO_POP', 'SERTANEJO_UNIVERSITARIO', 'SHIMMER_POP', 'SHIMMER_PSYCH', 'SHIVER_POP', 'SHOEGAZE', 'SHOW_TUNES', 'SINGAPOREAN_POP', 'SKATE_PUNK', 'SKWEEE', 'SKY_ROOM', 'SLAYER', 'SMALL_ROOM', 'SMOOTH_JAZZ', 'SMOOTH_SAXOPHONE', 'SOCAL_POP_PUNK', 'SOCIAL_MEDIA_POP', 'SOFT_ROCK', 'SOUL', 'SOUND_ART', 'SOUTHAMPTON_INDIE', 'SOUTHERN_HIP_HOP', 'SOUTH_AFRICAN_ALTERNATIVE', 'SOUTH_AFRICAN_HIP_HOP', 'SOUTH_AFRICAN_POP', 'SOUTH_AFRICAN_ROCK', 'SPANISH_HIP_HOP', 'SPANISH_INDIE_POP', 'SPANISH_NOISE_POP', 'SPANISH_POP', 'SPEED_GARAGE', 'STOCKHOLM_INDIE', 'STOMP_AND_HOLLER', 'SUBSTEP', 'SWEDISH_ALTERNATIVE_ROCK', 'SWEDISH_ELECTRONIC', 'SWEDISH_ELECTROPOP', 'SWEDISH_GANGSTA_RAP', 'SWEDISH_HIP_HOP', 'SWEDISH_IDOL_POP', 'SWEDISH_INDIE_ROCK', 'SWEDISH_JAZZ', 'SWEDISH_POP', 'SWEDISH_REGGAE', 'SWEDISH_SINGERSONGWRITER', 'SWEDISH_SOUL', 'SWEDISH_TECHNO', 'SWEDISH_TROPICAL_HOUSE', 'SWEDISH_URBAN', 'SWING', 'SWISS_INDIE', 'SYDNEY_INDIE', 'TALENT_SHOW', 'TECHNO', 'TECH_HOUSE', 'TEEN_POP', 'TEXAS_COUNTRY', 'TIJUANA_ELECTRONIC', 'TORONTO_INDIE', 'TORONTO_RAP', 'TRANCECORE', 'TRANSPOP', 'TRAP', 'TRAPRUN', 'TRAP_ARGENTINO', 'TRAP_BRASILEIRO', 'TRAP_CATALA', 'TRAP_CHILENO', 'TRAP_ESPANOL', 'TRAP_FRANCAIS', 'TRAP_LATINO', 'TRAP_QUEEN', 'TRAP_SOUL', 'TRIBAL_HOUSE', 'TROPICAL', 'TROPICAL_HOUSE', 'TURKISH_HIP_HOP', 'TURKISH_JAZZ', 'TURKISH_POP', 'UAE_INDIE', 'UKRAINIAN_POP', 'UK_ALTERNATIVE_HIP_HOP', 'UK_ALTERNATIVE_POP', 'UK_AMERICANA', 'UK_CONTEMPORARY_RB', 'UK_DANCE', 'UK_DANCEHALL', 'UK_DNB', 'UK_DRILL', 'UK_EXPERIMENTAL_ELECTRONIC', 'UK_FUNKY', 'UK_HIP_HOP', 'UK_HOUSE', 'UK_POP', 'UK_TECH_HOUSE', 'UNDERGROUND_HIP_HOP', 'UPLIFTING_TRANCE', 'URBAN_CONTEMPORARY', 'UTAH_INDIE', 'VANCOUVER_INDIE', 'VAPOR_HOUSE', 'VAPOR_POP', 'VAPOR_SOUL', 'VAPOR_TRAP', 'VAPOR_TWITCH', 'VENEZUELAN_HIP_HOP', 'VERACRUZ_INDIE', 'VERMONT_INDIE', 'VICTORIA_BC_INDIE', 'VIRAL_POP', 'VIRAL_TRAP', 'VOCAL_HOUSE', 'VOCAL_JAZZ', 'WARM_DRONE', 'WAVE', 'WELSH_INDIE', 'WEST_AUSTRALIAN_HIP_HOP', 'WEST_COAST_TRAP', 'WISCONSIN_INDIE', 'WITCH_HOUSE', 'WONKY', 'WORLD_FUSION', 'ZAPSTEP', 'ZIMDANCEHALL']

def read_file(tsv_file):
    tracks = {}
    with open(tsv_file) as fp:
        reader = csv.reader(fp, delimiter='\t')
        next(reader, None)  # skip header
        for row in reader:
            track_id = row[0]
            tracks[track_id] = {
                'path': row[3].replace('.mp3', '.npy').replace('.m4a', '.npy'),
                'tags': row[5:],
            }
    return tracks


class AudioFolder(data.Dataset):
    def __init__(self, root, split, input_length=None):
        self.root = root
        self.split = split
        self.input_length = input_length
        self.get_songlist()

    def __getitem__(self, index):
        npy, tag_binary = self.get_npy(index)
        return npy.astype('float32'), tag_binary.astype('float32')

    def get_songlist(self):
        self.mlb = LabelBinarizer().fit(TAGS)
        if self.split == 'TRAIN':
            train_file = os.path.join(META_PATH, 'train.tsv')
            self.file_dict = read_file(train_file)
            self.fl = list(self.file_dict.keys())
        elif self.split == 'VALID':
            train_file = os.path.join(META_PATH,'validation.tsv')
            self.file_dict= read_file(train_file)
            self.fl = list(self.file_dict.keys())
        elif self.split == 'TEST':
            test_file = os.path.join(META_PATH, 'test.tsv')
            self.file_dict= read_file(test_file)
            self.fl = list(self.file_dict.keys())
        else:
            print('Split should be one of [TRAIN, VALID, TEST]')


    def get_npy(self, index):
        jmid = self.fl[index]
        filename = self.file_dict[jmid]['path']
        npy_path = os.path.join(self.root, filename.split("/")[-1])
        npy = np.load(npy_path, mmap_mode='r')
        random_idx = int(np.floor(np.random.random(1) * (len(npy)-self.input_length)))
        npy = np.array(npy[random_idx:random_idx+self.input_length])
        tag_binary = np.sum(self.mlb.transform(self.file_dict[jmid]['tags']), axis=0)
        return npy, tag_binary

    def __len__(self):
        return len(self.fl)

def get_audio_loader(root, batch_size, split='TRAIN', num_workers=0, input_length=None):
    data_loader = data.DataLoader(dataset=AudioFolder(root, split=split, input_length=input_length),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=num_workers)
    return data_loader

