# machine-learning-challenge

Data cleanup

Remove rows (if any) where koi_pdisposition is not FALSE POSITIVE or CANDIDATE; koi_disposition has additional categories. None found in current file.

Drop error columns (although these could be useful in the real world), extra IDs, KOI score, and extra evaluation columns

Drop nans:
exoplanets_basic.dropna(axis=0)
#None found by this method

np.any(np.isnan(exoplanets_basic))
np.all(np.isfinite(exoplanets_basic))



exoplanet_search_KNN

KNN, all columns
data scaled, split for training
for k in range(1, 40, 2):
k=15 Test Acc: 0.987

KNN, take out "flag"values
for k in range(1, 50, 2):
k=17 Test Acc: 0.761


exoplanet_search_deep
data scaled, split for training

First run: 

model = Sequential()
model.add(Dense(units=100, activation='relu', input_dim=20))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=2, activation='softmax'))
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 100)               2100      
_________________________________________________________________
dense_1 (Dense)              (None, 100)               10100     
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 202       
=================================================================
Total params: 12,402
Trainable params: 12,402
Non-trainable params: 0
epochs=60,
shuffle=True
2300/1 - 1s - loss: 0.0306 - accuracy: 0.9887
Normal Neural Network 1 - Loss: 0.05442659353351464, Accuracy: 0.9886956810951233

Second run: Same parameters, but remove "flag" columns

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_3 (Dense)              (None, 100)               1600      
_________________________________________________________________
dense_4 (Dense)              (None, 100)               10100     
_________________________________________________________________
dense_5 (Dense)              (None, 2)                 202       
=================================================================
Total params: 11,902
Trainable params: 11,902
Non-trainable params: 0
2300/1 - 0s - loss: 0.3951 - accuracy: 0.8122
Normal Neural Network 2 - Loss: 0.43118020280547764, Accuracy: 0.8121739029884338


Third run: "flag" columns removed, added a layer
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_6 (Dense)              (None, 100)               1600      
_________________________________________________________________
dense_7 (Dense)              (None, 100)               10100     
_________________________________________________________________
dense_8 (Dense)              (None, 100)               10100     
_________________________________________________________________
dense_9 (Dense)              (None, 2)                 202       
=================================================================
Total params: 22,002
Trainable params: 22,002
Non-trainable params: 0
2300/1 - 1s - loss: 0.4006 - accuracy: 0.8113
Normal Neural Network 3 - Loss: 0.4283171708687492, Accuracy: 0.8113043308258057


Fourth run: Like third, but different starting state
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_10 (Dense)             (None, 100)               1600      
_________________________________________________________________
dense_11 (Dense)             (None, 100)               10100     
_________________________________________________________________
dense_12 (Dense)             (None, 100)               10100     
_________________________________________________________________
dense_13 (Dense)             (None, 2)                 202       
=================================================================
Total params: 22,002
Trainable params: 22,002
Non-trainable params: 0
2300/1 - 1s - loss: 0.4833 - accuracy: 0.8091
Normal Neural Network 4 - Loss: 0.4340094618175341, Accuracy: 0.8091304302215576


Fifth run: Like third, but twice the units
Model: "sequential_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_14 (Dense)             (None, 200)               3200      
_________________________________________________________________
dense_15 (Dense)             (None, 200)               40200     
_________________________________________________________________
dense_16 (Dense)             (None, 200)               40200     
_________________________________________________________________
dense_17 (Dense)             (None, 2)                 402       
=================================================================
Total params: 84,002
Trainable params: 84,002
Non-trainable params: 0
2300/1 - 1s - loss: 0.3742 - accuracy: 0.8117
Normal Neural Network 5 - Loss: 0.4426208177338476, Accuracy: 0.8117391467094421


Sixth run: Like third, but with twice the epochs
Model: "sequential_5"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_18 (Dense)             (None, 100)               1600      
_________________________________________________________________
dense_19 (Dense)             (None, 100)               10100     
_________________________________________________________________
dense_20 (Dense)             (None, 100)               10100     
_________________________________________________________________
dense_21 (Dense)             (None, 2)                 202       
=================================================================
Total params: 22,002
Trainable params: 22,002
Non-trainable params: 0
2300/1 - 1s - loss: 0.3631 - accuracy: 0.8104
Normal Neural Network 6 - Loss: 0.4818687497014585, Accuracy: 0.8104347586631775


Seventh run: same as third, but try "Nadam" normalization
Model: "sequential_6"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_22 (Dense)             (None, 100)               1600      
_________________________________________________________________
dense_23 (Dense)             (None, 100)               10100     
_________________________________________________________________
dense_24 (Dense)             (None, 100)               10100     
_________________________________________________________________
dense_25 (Dense)             (None, 2)                 202       
=================================================================
Total params: 22,002
Trainable params: 22,002
Non-trainable params: 0
2300/1 - 1s - loss: 0.3636 - accuracy: 0.8187
Normal Neural Network 7 - Loss: 0.42853991181954093, Accuracy: 0.8186956644058228


Eighth run: same as third, but try "Adamax" normalization
Model: "sequential_7"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_26 (Dense)             (None, 100)               1600      
_________________________________________________________________
dense_27 (Dense)             (None, 100)               10100     
_________________________________________________________________
dense_28 (Dense)             (None, 100)               10100     
_________________________________________________________________
dense_29 (Dense)             (None, 2)                 202       
=================================================================
Total params: 22,002
Trainable params: 22,002
Non-trainable params: 0
2300/1 - 1s - loss: 0.4071 - accuracy: 0.8122
Normal Neural Network 8 - Loss: 0.43720548977022583, Accuracy: 0.8121739029884338


Exoplanet_search_Decision_trees

First run: Decision Tree, random state=57, "flag" columns included
First run results: clf.score(X_test, y_test) 0.9843478260869565

Second run: Decision Tree, random state=57, "flag" variables removed
Second run results: clf.score(X_test, y_test) 0.7669565217391304


Exoplanet_search_Random_forest

Random Forest, all columns, random state = 57, n_estimators = 200
rf.score(X_test, y_test) 0.9904347826086957
[(0.22733560271085565, 'flag_not_transit_like'),
 (0.19420355973907696, 'flag_centroid_offset'),
 (0.1440095276751463, 'flag_stellar_eclipse'),
 (0.10109978958257873, 'planet_radius'),
 (0.07024011822169293, 'flag_ephemeris match'),
 (0.044640688652287444, 'stellar_flux_loss_at_trans_min'),
 (0.03311771350470804, 'orbital_period'),
 (0.031480649395637335, 'trans_sig_to_noise'),
 (0.030972109742308376, 'insolation_flux'),
 (0.029535382736840466, 'approx_planet_temp'),
 (0.020736500860786416, 'star_planet_dist_at_conj'),
 (0.018509642109181546, 'trans_duration'),
 (0.013508461271531618, 'time_first_trans_detected'),
 (0.008068372003769954, 'stellar_eff_temp'),
 (0.007187249648553146, 'stellar_photosph_rad'),
 (0.00658100063113589, 'sky_location_right_asc'),
 (0.006470822163761753, 'stellar_surf_gravity'),
 (0.006287781229770447, 'stellar_magnitude'),
 (0.006015028120377005, 'sky_location_declination')]


Random forest with "flag" variables removed; random state=57
rf.score(X_test, y_test)  1.0
[(0.0756992214210558, 'planet_radius'),
 (0.02594718491892835, 'stellar_flux_loss_at_trans_min'),
 (0.025467720100393297, 'insolation_flux'),
 (0.022833793544298753, 'star_planet_dist_at_conj'),
 (0.02201414443349766, 'orbital_period'),
 (0.020332680056512346, 'approx_planet_temp'),
 (0.019891447901408545, 'trans_sig_to_noise'),
 (0.012684390468460921, 'trans_duration'),
 (0.007709256710770052, 'time_first_trans_detected'),
 (0.0032457954749037054, 'stellar_eff_temp'),
 (0.002960467272649715, 'stellar_photosph_rad'),
 (0.002777782158230879, 'sky_location_right_asc'),
 (0.002364284916553437, 'stellar_surf_gravity'),
 (0.0018585959312780007, 'sky_location_declination'),
 (0.001716789180097814, 'stellar_magnitude')]

Random forest, all columns; random state=57; n_estimators=50
rf.score(X_test, y_test) 0.9895652173913043
[(0.0756992214210558, 'planet_radius'),
 (0.02594718491892835, 'stellar_flux_loss_at_trans_min'),
 (0.025467720100393297, 'insolation_flux'),
 (0.022833793544298753, 'star_planet_dist_at_conj'),
 (0.02201414443349766, 'orbital_period'),
 (0.020332680056512346, 'approx_planet_temp'),
 (0.019891447901408545, 'trans_sig_to_noise'),
 (0.012684390468460921, 'trans_duration'),
 (0.007709256710770052, 'time_first_trans_detected'),
 (0.0032457954749037054, 'stellar_eff_temp'),
 (0.002960467272649715, 'stellar_photosph_rad'),
 (0.002777782158230879, 'sky_location_right_asc'),
 (0.002364284916553437, 'stellar_surf_gravity'),
 (0.0018585959312780007, 'sky_location_declination'),
 (0.001716789180097814, 'stellar_magnitude')]


Random forest , all columns; random state=57; n_estimators=10
rf.score(X_test, y_test) 0.9891304347826086
[(0.2259593552429587, 'flag_not_transit_like'),
 (0.20216741873290553, 'flag_centroid_offset'),
 (0.1376671830280171, 'flag_stellar_eclipse'),
 (0.12487004832135692, 'planet_radius'),
 (0.07077921338368869, 'flag_ephemeris match'),
 (0.03676421924755866, 'stellar_flux_loss_at_trans_min'),
 (0.031613473252989584, 'orbital_period'),
 (0.029247794125857682, 'insolation_flux'),
 (0.027746302416463732, 'approx_planet_temp'),
 (0.027118406419649455, 'trans_sig_to_noise'),
 (0.017319409129850877, 'trans_duration'),
 (0.016328628721901377, 'star_planet_dist_at_conj'),
 (0.013032099322464014, 'time_first_trans_detected'),
 (0.007692441245159498, 'stellar_eff_temp'),
 (0.007656213060912029, 'stellar_surf_gravity'),
 (0.006604557561888076, 'stellar_photosph_rad'),
 (0.006000789138146452, 'sky_location_right_asc'),
 (0.005907446209645676, 'sky_location_declination'),
 (0.005525001438586003, 'stellar_magnitude')]

Random Forest, all columns, random state = 312, n_estimators = 200
rf.score(X_test, y_test) 0.9882608695652174
[(0.23798774600917552, 'flag_not_transit_like'),
 (0.19186263353584754, 'flag_centroid_offset'),
 (0.16692955018983197, 'flag_stellar_eclipse'),
 (0.08542584938692699, 'planet_radius'),
 (0.07186881412337937, 'flag_ephemeris match'),
 (0.04271948290966886, 'stellar_flux_loss_at_trans_min'),
 (0.03130938860531501, 'insolation_flux'),
 (0.029941578261044405, 'trans_sig_to_noise'),
 (0.029162560264454945, 'orbital_period'),
 (0.028171703218690346, 'approx_planet_temp'),
 (0.01932885503213915, 'star_planet_dist_at_conj'),
 (0.015147970719536408, 'trans_duration'),
 (0.01031685362043467, 'time_first_trans_detected'),
 (0.007974241401576123, 'stellar_eff_temp'),
 (0.007710574883976139, 'stellar_photosph_rad'),
 (0.006595751769123111, 'stellar_surf_gravity'),
 (0.0061432410779326515, 'sky_location_right_asc'),
 (0.005783169289187998, 'stellar_magnitude'),
 (0.005620035701758849, 'sky_location_declination')]


Random Forest, all columns, random state = 57, n_estimators = 200, criterion = "entropy"
rf.score(X_test, y_test) 0.9904347826086957
[(0.22605002898133872, 'flag_not_transit_like'),
 (0.19519889356462308, 'flag_centroid_offset'),
 (0.15197875446292186, 'flag_stellar_eclipse'),
 (0.0838030500923313, 'planet_radius'),
 (0.07276204231577676, 'flag_ephemeris match'),
 (0.04219325150244801, 'stellar_flux_loss_at_trans_min'),
 (0.03658629069697878, 'trans_sig_to_noise'),
 (0.03450634261202376, 'orbital_period'),
 (0.029405794268847978, 'insolation_flux'),
 (0.025164503624640545, 'approx_planet_temp'),
 (0.020040278368036304, 'star_planet_dist_at_conj'),
 (0.018600221393501714, 'trans_duration'),
 (0.013328520799608404, 'time_first_trans_detected'),
 (0.009693009092842604, 'stellar_photosph_rad'),
 (0.009514958668370243, 'stellar_eff_temp'),
 (0.008082442537119995, 'stellar_surf_gravity'),
 (0.007957083474579353, 'sky_location_right_asc'),
 (0.007719477673848345, 'stellar_magnitude'),
 (0.007415055870162336, 'sky_location_declination')]


Random Forest, all columns, random state = 57, n_estimators = 200, max_features='auto'
rf.score(X_test, y_test) 0.9904347826086957
[(0.22872426067052315, 'flag_not_transit_like'),
 (0.19127100946869177, 'flag_centroid_offset'),
 (0.16299930629129977, 'flag_stellar_eclipse'),
 (0.08718918447587949, 'planet_radius'),
 (0.07243404191102386, 'flag_ephemeris match'),
 (0.0373124679610056, 'trans_sig_to_noise'),
 (0.03577251639510632, 'stellar_flux_loss_at_trans_min'),
 (0.035374110557582196, 'orbital_period'),
 (0.02977581241967809, 'approx_planet_temp'),
 (0.026619083006408518, 'insolation_flux'),
 (0.021057379335761735, 'star_planet_dist_at_conj'),
 (0.01629170482114803, 'trans_duration'),
 (0.013611690283884705, 'time_first_trans_detected'),
 (0.008348201095845754, 'stellar_eff_temp'),
 (0.007934720066254973, 'stellar_photosph_rad'),
 (0.006717208390302393, 'stellar_surf_gravity'),
 (0.006539054804130948, 'sky_location_right_asc'),
 (0.006139290905798041, 'stellar_magnitude'),
 (0.005888957139674693, 'sky_location_declination')]

Random forest with "flag" variables removed; random state = 57, n_estimators = 200, criterion = "entropy"
rf.score(X_test, y_test) 0.8386956521739131
[(0.1379310462689551, 'planet_radius'),
 (0.09258523408807015, 'orbital_period'),
 (0.08848849033703969, 'stellar_flux_loss_at_trans_min'),
 (0.08450236297953793, 'trans_sig_to_noise'),
 (0.07993781905557479, 'trans_duration'),
 (0.07773010302285813, 'star_planet_dist_at_conj'),
 (0.06802682933552859, 'insolation_flux'),
 (0.06287145573790465, 'approx_planet_temp'),
 (0.0522589495016858, 'time_first_trans_detected'),
 (0.04521993048295201, 'sky_location_right_asc'),
 (0.043680510036280915, 'stellar_eff_temp'),
 (0.0434146383951555, 'stellar_surf_gravity'),
 (0.043329969186600946, 'stellar_photosph_rad'),
 (0.04102089876256105, 'sky_location_declination'),
 (0.03900176280929468, 'stellar_magnitude')]

Random forest with "flag" variables removed; random state = 57, n_estimators = 200, max_features='auto'
rf.score(X_test, y_test) 0.8369565217391305
[(0.14776291108077555, 'planet_radius'),
 (0.09561514385817545, 'orbital_period'),
 (0.08686027002282865, 'trans_sig_to_noise'),
 (0.08370918639742753, 'star_planet_dist_at_conj'),
 (0.0809737656439492, 'stellar_flux_loss_at_trans_min'),
 (0.0768699129102505, 'trans_duration'),
 (0.07343482879377032, 'approx_planet_temp'),
 (0.07113635596863883, 'insolation_flux'),
 (0.05086760639057528, 'time_first_trans_detected'),
 (0.041273752166773474, 'stellar_eff_temp'),
 (0.041271588512042455, 'sky_location_right_asc'),
 (0.039728371705770225, 'stellar_photosph_rad'),
 (0.038560686609861984, 'stellar_surf_gravity'),
 (0.036885649669834576, 'sky_location_declination'),
 (0.035049970269326075, 'stellar_magnitude')]


































