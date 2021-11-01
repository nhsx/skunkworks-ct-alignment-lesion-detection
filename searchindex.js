Search.setIndex({docnames:["ai_ct_scans","ai_ct_scans.GUI","ai_ct_scans.GUI.image_viewer","ai_ct_scans.GUI.scan_viewer","ai_ct_scans.GUI.ui_stack_window","ai_ct_scans.GUI.viewer_widget","ai_ct_scans.data_loading","ai_ct_scans.data_writing","ai_ct_scans.image_processing_utils","ai_ct_scans.keypoint_alignment","ai_ct_scans.model_trainers","ai_ct_scans.models","ai_ct_scans.non_rigid_alignment","ai_ct_scans.phase_correlation","ai_ct_scans.phase_correlation_image_processing","ai_ct_scans.point_matching","ai_ct_scans.scan_tool","ai_ct_scans.sectioning","index","modules","readme"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["ai_ct_scans.rst","ai_ct_scans.GUI.rst","ai_ct_scans.GUI.image_viewer.rst","ai_ct_scans.GUI.scan_viewer.rst","ai_ct_scans.GUI.ui_stack_window.rst","ai_ct_scans.GUI.viewer_widget.rst","ai_ct_scans.data_loading.rst","ai_ct_scans.data_writing.rst","ai_ct_scans.image_processing_utils.rst","ai_ct_scans.keypoint_alignment.rst","ai_ct_scans.model_trainers.rst","ai_ct_scans.models.rst","ai_ct_scans.non_rigid_alignment.rst","ai_ct_scans.phase_correlation.rst","ai_ct_scans.phase_correlation_image_processing.rst","ai_ct_scans.point_matching.rst","ai_ct_scans.scan_tool.rst","ai_ct_scans.sectioning.rst","index.rst","modules.rst","readme.rst"],objects:{"":{ai_ct_scans:[0,0,0,"-"]},"ai_ct_scans.GUI":{image_viewer:[2,0,0,"-"],scan_viewer:[3,0,0,"-"],ui_stack_window:[4,0,0,"-"],viewer_widget:[5,0,0,"-"]},"ai_ct_scans.GUI.image_viewer":{ImageViewer:[2,1,1,""]},"ai_ct_scans.GUI.image_viewer.ImageViewer":{paintEvent:[2,2,1,""],painter_element:[2,2,1,""],pixmap:[2,3,1,""],scale_factor:[2,2,1,""],set_image:[2,2,1,""],staticMetaObject:[2,3,1,""]},"ai_ct_scans.GUI.scan_viewer":{ScanViewer:[3,1,1,""]},"ai_ct_scans.GUI.scan_viewer.ScanViewer":{Ui:[3,1,1,""],convert_for_viewer:[3,2,1,""],display_data:[3,2,1,""],pre_process_data:[3,2,1,""],set_data:[3,2,1,""],staticMetaObject:[3,3,1,""],viewer_data:[3,3,1,""],volume_plot:[3,3,1,""]},"ai_ct_scans.GUI.ui_stack_window":{UiStackWindow:[4,1,1,""]},"ai_ct_scans.GUI.ui_stack_window.UiStackWindow":{Ui:[4,1,1,""],choose_data_dir:[4,2,1,""],data_path:[4,3,1,""],load_data:[4,2,1,""],staticMetaObject:[4,3,1,""],ui:[4,3,1,""]},"ai_ct_scans.GUI.viewer_widget":{SliceDirection:[5,1,1,""],ViewerWidget:[5,1,1,""]},"ai_ct_scans.GUI.viewer_widget.SliceDirection":{AXIAL:[5,3,1,""],CORONAL:[5,3,1,""],SAGITTAL:[5,3,1,""]},"ai_ct_scans.GUI.viewer_widget.ViewerWidget":{Ui:[5,1,1,""],blank_frames:[5,3,1,""],calculate_non_rigid_alignment:[5,2,1,""],change_orientation:[5,2,1,""],cpd_aligned_scan:[5,3,1,""],data:[5,3,1,""],detect_ellipses:[5,2,1,""],display_results:[5,2,1,""],display_sectioned_slice:[5,2,1,""],display_slice:[5,2,1,""],extract_centre_coordinates_from_string:[5,2,1,""],extract_ellipse_overlay_info:[5,2,1,""],handle_item_selection:[5,2,1,""],local_region:[5,3,1,""],orientation:[5,3,1,""],phase_correlation_shift:[5,3,1,""],set_data:[5,2,1,""],setup_orientation:[5,2,1,""],show_phase_correlation:[5,2,1,""],staticMetaObject:[5,3,1,""],toggle_ellipse_table:[5,2,1,""],toggle_ellipse_table_button:[5,2,1,""],toggle_non_rigid_2d_controls:[5,2,1,""],toggle_phase_correlation_controls:[5,2,1,""],toggle_tissue_sectioning:[5,2,1,""],toggle_warp_display:[5,2,1,""],ui:[5,3,1,""],update_ellipse_table:[5,2,1,""],update_local_region:[5,2,1,""],update_scan:[5,2,1,""],update_scan_toggle:[5,2,1,""]},"ai_ct_scans.data_loading":{BodyPartLoader:[6,1,1,""],MultiPatientAxialStreamer:[6,1,1,""],MultiPatientLoader:[6,1,1,""],PatientLoader:[6,1,1,""],ScanLoader:[6,1,1,""],data_root_directory:[6,4,1,""],dir_dicom_paths:[6,4,1,""],load_memmap:[6,4,1,""],load_validation_set:[6,4,1,""]},"ai_ct_scans.data_loading.BodyPartLoader":{scan_1:[6,3,1,""],scan_2:[6,3,1,""]},"ai_ct_scans.data_loading.MultiPatientAxialStreamer":{reset_indices:[6,2,1,""],stream_next:[6,2,1,""]},"ai_ct_scans.data_loading.MultiPatientLoader":{patient_paths:[6,3,1,""],patients:[6,3,1,""],root_dir:[6,3,1,""]},"ai_ct_scans.data_loading.PatientLoader":{abdo:[6,3,1,""]},"ai_ct_scans.data_loading.ScanLoader":{clear_scan:[6,2,1,""],delete_memmap:[6,2,1,""],dicom_paths:[6,3,1,""],full_scan:[6,3,1,""],full_scan_to_memmap:[6,2,1,""],load_2d_array:[6,2,1,""],load_full_memmap:[6,2,1,""],load_memmap_and_clear_scan:[6,2,1,""],load_patient_metadata:[6,2,1,""],load_scan:[6,2,1,""],mean_axial_thickness:[6,2,1,""],raw_transverse_pixel_spacing_and_shape:[6,2,1,""],rescale_depth:[6,2,1,""]},"ai_ct_scans.data_writing":{create_dicom_file:[7,4,1,""],ndarray_to_memmap:[7,4,1,""]},"ai_ct_scans.image_processing_utils":{normalise:[8,4,1,""],overlay_warp_on_slice:[8,4,1,""]},"ai_ct_scans.keypoint_alignment":{align_image:[9,4,1,""],find_homography:[9,4,1,""],get_keypoints_and_descriptors:[9,4,1,""],match_descriptors:[9,4,1,""],sieve_matches_lowe:[9,4,1,""]},"ai_ct_scans.model_trainers":{InfillTrainer:[10,1,1,""],blur:[10,4,1,""],debug_plot:[10,4,1,""],det:[10,4,1,""]},"ai_ct_scans.model_trainers.InfillTrainer":{border_mask_builder:[10,2,1,""],build_batch:[10,2,1,""],load_model:[10,2,1,""],loss:[10,2,1,""],plane_mask_builder:[10,2,1,""],process_full_scan:[10,2,1,""],random_axial_slicer:[10,2,1,""],random_coronal_slicer:[10,2,1,""],random_sagittal_slicer:[10,2,1,""],save_model:[10,2,1,""],train_for_iterations:[10,2,1,""],train_step:[10,2,1,""]},"ai_ct_scans.models":{Infiller:[11,1,1,""],SingleDecoderLayer:[11,1,1,""],SingleEncoderLayer:[11,1,1,""]},"ai_ct_scans.models.Infiller":{build_decoder_convs:[11,2,1,""],build_encoder_convs:[11,2,1,""],build_latent_space_bridge:[11,2,1,""],forward:[11,2,1,""],training:[11,3,1,""]},"ai_ct_scans.models.SingleDecoderLayer":{forward:[11,2,1,""],training:[11,3,1,""]},"ai_ct_scans.models.SingleEncoderLayer":{forward:[11,2,1,""],training:[11,3,1,""]},"ai_ct_scans.non_rigid_alignment":{align_2D_using_CPD:[12,4,1,""],align_3D_using_CPD:[12,4,1,""],estimate_3D_alignment_transform:[12,4,1,""],get_warp_overlay:[12,4,1,""],main:[12,4,1,""],read_transform:[12,4,1,""],transform_3d_volume:[12,4,1,""],transform_3d_volume_in_chunks:[12,4,1,""],write_transform:[12,4,1,""]},"ai_ct_scans.phase_correlation":{align_via_phase_correlation_2d:[13,4,1,""],find_shift_via_phase_correlation_2d:[13,4,1,""],shift_image_2d:[13,4,1,""],shift_nd:[13,4,1,""],shift_via_phase_correlation_nd:[13,4,1,""],shifts_via_local_region:[13,4,1,""]},"ai_ct_scans.phase_correlation_image_processing":{circle:[14,4,1,""],convolve:[14,4,1,""],generate_overlay_2d:[14,4,1,""],lmr:[14,4,1,""],max_shape_from_image_list:[14,4,1,""],pad_nd:[14,4,1,""],sphere:[14,4,1,""],zero_crossings:[14,4,1,""]},"ai_ct_scans.point_matching":{abs_dist_cost_matrix:[15,4,1,""],match_indices:[15,4,1,""]},"ai_ct_scans.scan_tool":{MainWindow:[16,1,1,""],main:[16,4,1,""]},"ai_ct_scans.scan_tool.MainWindow":{Ui:[16,1,1,""],staticMetaObject:[16,3,1,""],ui:[16,3,1,""]},"ai_ct_scans.sectioning":{CTEllipsoidFitter:[17,1,1,""],DinoSectioner:[17,1,1,""],EllipseFitter:[17,1,1,""],HierarchicalMeanShift:[17,1,1,""],MeanShiftWithProbs:[17,1,1,""],TextonSectioner:[17,1,1,""]},"ai_ct_scans.sectioning.CTEllipsoidFitter":{draw_ellipses_2d:[17,2,1,""],draw_ellipsoid_walls:[17,2,1,""],find_ellipsoids:[17,2,1,""]},"ai_ct_scans.sectioning.DinoSectioner":{load:[17,2,1,""],load_dino_model:[17,2,1,""],save:[17,2,1,""],single_image_texton_descriptors:[17,2,1,""]},"ai_ct_scans.sectioning.EllipseFitter":{fit_ellipses:[17,2,1,""]},"ai_ct_scans.sectioning.HierarchicalMeanShift":{fit:[17,2,1,""],predict:[17,2,1,""],predict_full:[17,2,1,""],predict_proba:[17,2,1,""],predict_proba_secondary:[17,2,1,""],predict_secondary:[17,2,1,""]},"ai_ct_scans.sectioning.MeanShiftWithProbs":{fit:[17,2,1,""],predict_proba:[17,2,1,""]},"ai_ct_scans.sectioning.TextonSectioner":{build_sample_texton_set:[17,2,1,""],label_im:[17,2,1,""],load:[17,2,1,""],probabilities_im:[17,2,1,""],save:[17,2,1,""],single_image_texton_descriptors:[17,2,1,""],train_clusterers:[17,2,1,""]},ai_ct_scans:{GUI:[1,0,0,"-"],data_loading:[6,0,0,"-"],data_writing:[7,0,0,"-"],image_processing_utils:[8,0,0,"-"],keypoint_alignment:[9,0,0,"-"],model_trainers:[10,0,0,"-"],models:[11,0,0,"-"],non_rigid_alignment:[12,0,0,"-"],phase_correlation:[13,0,0,"-"],phase_correlation_image_processing:[14,0,0,"-"],point_matching:[15,0,0,"-"],scan_tool:[16,0,0,"-"],sectioning:[17,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function"},terms:{"0":[3,5,6,7,10,12,13,14,15,17,20],"05":10,"07":20,"0th":10,"1":[5,6,10,12,15,17,20],"10":[10,11,12,17,20],"100":20,"1000":7,"10000":17,"100000":17,"10k":20,"12":20,"14519":20,"15":20,"16":17,"161200771652309664842225892934":20,"180":20,"1d":[10,13,17],"1e":10,"1s":17,"1st":10,"2":[5,6,10,13,17,20],"20":20,"200":10,"2000":20,"2002":20,"2018":20,"2021":20,"24":10,"25":17,"25000":17,"255":17,"256":[10,11],"256x256":10,"2d":[5,6,7,8,9,10,12,13,14,17,20],"2nd":10,"2s":17,"3":[5,10,11,13,17,20],"30000":12,"3273":20,"3320":20,"3d":[3,5,6,10,12,14,17,20],"3s":17,"4":20,"452":20,"456":20,"4628":17,"469":20,"4d":11,"5":[17,20],"50":17,"500":[6,17],"5000":[12,17,20],"50k":20,"512":[10,11],"5gb":20,"5mm":20,"6":[10,20],"64":10,"7":20,"75":17,"7mm":20,"8":[8,10,20],"8gb":10,"9":20,"9gb":20,"boolean":14,"case":20,"class":[2,3,4,5,6,10,11,16,17],"default":[6,10,14,17],"do":[6,10,17,20],"enum":5,"export":20,"float":[2,5,6,7,12,13,14,17],"function":[8,9,10,11,12],"int":[3,5,6,10,12,13,14,15,17],"new":[4,10,17],"return":[2,3,5,6,7,8,9,10,11,12,13,14,15,17],"static":3,"switch":[4,6],"true":[6,10,11,13,14,17],"while":[10,11,13,14],A:[5,6,8,9,10,11,12,13,14,15,17,20],As:20,BY:20,By:17,For:20,IN:6,If:[4,5,6,10,13,14,17,20],In:20,It:20,No:20,Not:11,The:[6,7,9,10,11,12,13,14,15,17,20],There:20,These:[17,20],To:20,_0:13,_:[6,20],__:6,_ellipses_to_rich_inform:17,_mean_shift:17,abdo1:20,abdo2:20,abdo:6,abdomen:[6,10,20],abil:20,abl:20,about:20,abov:[6,10,20],abs_dist_cost_matrix:15,absolut:[7,12],accept:[6,17],access:[7,10,20],accord:[6,17],account:20,accur:20,achiev:10,across:[13,14],act:20,ad:[3,14,20],add:[3,20],addit:[11,18,20],additional_librari:20,address:20,adjac:[13,14],affin:20,after:[10,11,12,13,17,20],afterward:11,against:17,ai_ct_scan:[18,20],aid:17,aim:20,air:[6,17],aiskunkwork:20,algorithm:[17,20],alias:10,align:[4,5,8,9,12,13],align_2d_using_cpd:12,align_3d_using_cpd:12,align_imag:9,align_via_phase_correlation_2d:13,aligned_slic:8,all:[6,10,11,13,14,17,20],allow:[6,7,10,11,20],allow_off_edg:10,along:[6,10,14],also:[17,20],although:11,amount:[13,14],an:[5,6,7,8,9,10,12,13,14,17,20],analysi:20,anatom:20,anchor:10,ani:[5,6,10,17,20],anomali:[10,20],anomyis:20,anoth:9,apidoc:20,appear:20,append:10,appl:13,appli:[3,9,10,12,13,14,17,20],applic:[4,16],apply_lmr:13,apply_zero_cross:13,approach:20,april:20,ar:[6,10,12,15,17,20],arbitrari:13,arch:17,architectur:20,archiv:20,area:[10,20],arg:6,argument:20,around:[5,10,13,17,20],arr:[7,10,17],arrai:[2,3,5,6,7,8,10,14,17],array_s:14,arteri:20,artifici:20,assess:20,assign:[5,14,15,17],associ:17,assumpt:20,attempt:20,attribut:[5,20],augment:3,author:20,auto:[13,14],autom:20,automat:[2,20],avail:20,averag:[6,10],ax:[5,6,17],axi:[5,10,14,17],axial:[5,6,7,10,12,17,20],axial_width:10,axis_ellipses_count:17,b:20,back:6,background:[14,17,18],background_v:17,backward:10,base:[2,3,4,5,6,10,11,16,17,20],base_cluster:17,bash:20,batch:[10,11],batch_height:10,batch_siz:10,batch_width:10,been:[5,12,17,20],befor:13,begin:[6,10],behaviour:20,being:13,belong:17,below:[6,17,20],best:[10,20],between:[4,6,13,15,17,20],bilateralfilt:10,bin:[13,14],binari:14,bit:8,blank:[5,10,17],blank_fram:5,blank_width:10,blur:10,blur_kernel:[10,17],bmatrix:10,bodi:[6,10,11,20],body_part:6,body_part_index:10,body_part_indic:10,bodypartload:6,bool:[3,6,10,11,13,14,17],border:[10,13],border_mask_build:10,border_wrap:10,bordertyp:10,both:[13,14,20],bound:17,boundari:17,box:[10,17],boxfilt:10,bridg:11,brief:10,broadli:20,build:[6,10,11,12,17,20],build_batch:[10,11],build_decoder_conv:11,build_encoder_conv:11,build_latent_space_bridg:11,build_sample_texton_set:17,builder:20,built:[10,11,17,20],button:5,bypass_loss_check:10,c274:18,calcul:[5,12,20],calculate_non_rigid_align:5,call:[10,11],callabl:12,can:[10,13,14,17,20],cancer:20,candid:20,capabl:20,capapbl:11,captur:20,carcinoma:20,care:[11,20],cart:20,caus:10,cc:20,cd:20,cdot:10,center:10,centr:[5,10,13,14,17,20],central:[10,11,13],chang:[2,5,12,20],change_orient:5,channel:[10,14],check:[6,20],checkpoint:20,checkpoint_kei:17,choos:[13,14],choose_data_dir:4,chosen:10,chunk:12,chunk_thick:12,ci:20,circl:[13,14],classic:20,clear:6,clear_previous_memmap:10,clear_scan:6,cli:[12,20],clinic:20,clone:20,close:17,closest:[5,15],cluster:[17,20],cluster_label:17,clusterer_ind:17,clusterer_titl:17,cnn:10,code:[17,20],codebas:20,coher:[12,20],col:[13,14],collect:20,colour:3,column:[10,14],com:20,combin:[3,20],come:9,command:20,commis:18,commit:20,common:20,commun:20,compar:[10,17,18,20],compat:20,complet:[18,20],complianc:20,comput:[9,11,20],computation:20,concept:20,conceptu:20,concern:20,configur:5,connect:11,consid:17,consortium:20,contain:[4,5,8,15,16,20],contenttim:6,contrast:20,contribut:[5,10,17],control:5,conv_bia:11,convent:[6,17],convers:20,convert:[2,3,17],convert_for_view:3,convolut:[11,14,17],convolv:14,coordin:[2,5,10,12,13],coords_input_arrai:10,copi:20,copyright:20,corner:[2,10],coron:[5,10,17],coronal_width:10,corpu:20,correct:[5,20],correect:2,correl:[5,13,20],correspond:[9,10],cost:15,could:20,count:[13,14],cours:20,cover:[10,18,20],covert:3,cpd:[4,5],cpd_align:5,cpd_aligned_scan:5,cptac:20,cpu:10,creat:[2,6,11,13],create_dicom_fil:7,create_png_directori:20,create_png_directory_large_random:20,creativ:20,crop:10,cross:[3,13,14],crown:20,ct:[6,17],ctcopi:20,ctellipsoidfitt:17,cuda:20,cudnn:20,current:[2,3,5,10,11,20],custom:[10,20],cv2:[9,17],cv_16:10,cv_16u:10,cv_32f:10,cv_64f:10,cv_8u:10,cycl:17,cycpd:20,data:[3,4,5,6,7,10,14,17,18],data_directori:6,data_load:[0,10,19],data_path:4,data_root_directori:6,data_writ:[0,19],dataset:[6,7,17,20],date:20,debug:10,debug_plot:10,decod:11,decoder_filts_per_lay:[10,11],decreas:11,dedic:20,deep:20,defin:[5,10,11,12,13,14,20],delete_memmap:6,deliveri:18,demonstr:20,depend:[5,20],deploi:20,deploy:20,depth:[6,10,13],describ:[2,5,10],descriptor:[9,17],descriptors_1:9,descriptors_2:9,desir:20,destin:9,det:10,detach:10,detail:20,detect:[5,9,13,17],detect_ellips:5,detector:9,deterior:20,develop:18,devic:20,diagnosi:20,diagram:20,dialog:4,dicom:[6,7,20],dicom_path:6,dicom_test_data_gener:20,dict:[2,3,5,10,11,17],dictionari:[2,3,5,10],did:20,differ:[6,13,14,20],difficulti:20,dimens:[13,14,15,17],dimension:[13,14],dimension_4:13,dino:17,dinosection:17,dir_dicom_path:6,direct:12,directli:20,directori:[4,6,7,10,20],discov:[13,17],discuss:20,disk:[12,20],displai:[2,3,5,20],display_data:3,display_result:5,display_sectioned_slic:5,display_slic:5,disrupt:20,distanc:[12,15,20],distinct:17,divid:14,dl:[6,20],dmatch:9,doc:20,docker:20,document:18,doe:20,doesn:6,domin:17,down:17,download:20,dpia:20,draw:17,draw_ellipses_2d:17,draw_ellipsoid_wal:17,drawn:17,drift:[12,20],drive:7,driven:20,dst:10,dtype:6,due:20,dure:[6,10,12,20],e:[5,11,13,17,20],each:[3,5,6,8,10,11,12,13,14,15,17,20],earli:20,eas:10,edg:[10,13,14,17],effect:[10,13],either:[5,6,13,20],element:[4,5,6,10,14,15,16,17,20],eliot:20,ellips:[2,5,17],ellipsdoid:20,ellipsefitt:17,ellipsoid:[5,17,20],ellipsoid_volum:17,ellipt:17,embed:6,empti:14,en:20,enabl:[6,17,20],encod:11,encoder_filts_per_lay:[10,11],end:[10,14,20],endometri:20,endpoint:[16,20],enough:14,ensur:[10,20],enter:20,entir:[6,10],environ:20,equal:14,equival:10,error:[10,20],estim:[9,12],estimate_3d_alignment_transform:12,etc:[10,17],euclidean:15,event:[2,20],everi:[10,11,14],exact:20,exampl:20,exclud:6,executepreprocessor:20,exist:[2,6,20],exit:20,expand:20,expect:[6,13,14,17,20],explain:20,explor:20,exponenti:11,extens:[7,20],extra_data:20,extract:[5,9,10,12,13,20],extract_centre_coordinates_from_str:5,extract_ellipse_overlay_info:5,extrapol:10,f:[10,20],facebook:20,facebookresearch:20,factor:[2,20],fail:17,fals:[3,6,10,14,17],fast:20,faster:6,featur:20,feed:17,file:[4,6,7,10,17,20],file_path:7,filenam:[6,7,10],fill:[13,14],filter:[10,11,12,13,14,17,20],filter_dist:12,filter_kernel:17,filter_typ:[14,17],find:[2,13,14,17,20],find_ellipsoid:17,find_homographi:9,find_shift_via_phase_correlation_2d:13,fine:10,first:[6,9,14,17,20],fit:[2,14,17],fit_ellips:17,fitellips:17,fitter:17,fixtur:20,flag:3,float64:6,flow:20,focuss:20,folder:[6,20],follow:[12,17,20],form:[5,6,10],format:[3,5],former:[6,11],forward:[10,11],found:[6,10,17,20],frac:10,fraction:12,frame:5,frequent:20,from:[5,6,8,9,10,12,13,14,15,17,20],full:[3,5,6,12,17,20],full_scan:6,full_scan_to_memmap:6,full_sub_structur:17,further:20,futur:[11,20],g:[5,11,14,17,20],gattia:20,gaussianblur:10,gdpr:20,gener:[7,8,12,14,15,17],generalis:20,generate_overlay_2d:14,generate_test_fil:20,geometr:12,georg:20,get:[6,9,10,13,14,15,17,18],get_keypoints_and_descriptor:9,get_warp_overlai:12,git:20,github:20,given:[6,20],glvolumeitem:3,good:9,good_match:9,govern:20,gpu:20,gpu_avail:20,graphic:20,grayscal:[14,17],greater:14,greyscal:[8,9,12],growth:20,gui:[0,16,19],guid:[17,20],h:20,ha:[5,6,10,12,17,20],handl:[5,20],handle_item_select:5,happen:20,happi:20,hard:[7,20],have:[6,9,10,11,17,20],hdotsfor:10,health:20,hear:20,heatmap:[8,12],height:10,help:[10,20],henc:17,here:[17,20],hide:5,hierarch:17,hierarchical_mean_shift_tissue_sectioner_model:20,hierarchicalmeanshift:17,hiererach:17,highlight:20,histogram:[13,14],hook:11,hospit:20,hot:10,how:[10,20],howev:20,html:20,http:20,hub:20,i0:[6,20],i1:20,i30f:20,i:[6,13,15,17],id:20,idea:20,identifi:[5,20],ie:10,ignor:[6,11],ignore_rescal:6,ignore_z_rescal:6,im:17,imag:[2,6,8,9,10,11,12,13,14,17,20],image_1:[13,14],image_2:[13,14],image_processing_util:[0,19],image_view:[0,1],imageview:2,impact:20,implement:[11,20],improv:20,includ:[10,11,13,20],incompat:20,increas:11,independ:10,index:[3,5,6,10,15,17,18,20],indic:[6,10,15,17],indici:15,individu:20,inf:17,infil:[10,11],infiller_with_blur:20,infilltrain:[10,11],info_str:5,info_text:2,inform:[2,7,20],ingest:20,initi:20,initialis:20,inlier:9,inlin:20,inplac:20,input:[10,11,12,14,20],input_body_part:10,input_coord:10,input_height:11,input_imag:[2,10,11],input_plan:10,input_width:11,insert:5,inspect:20,inst:12,instanc:[2,4,11],instanti:17,instead:[11,20],institut:20,instruct:20,integ:[15,17],integr:[17,20],intellig:20,intend:18,intens:[8,12,17,20],interest:[13,20],interfac:20,intern:[4,17],intuit:6,investig:20,io:20,isn:6,item:[3,5],iter:10,its:[6,14,20],itself:17,j:15,join:20,jul:20,jun:20,jupyt:20,k:10,kernel:[10,14,17],kernel_s:[10,11],key_points_1:9,key_points_2:9,keypoint:[9,20],keypoint_align:[0,19],known:[17,18],ksize:10,kwarg:17,label:[3,5,10,17],label_im:17,labour:20,lack:20,larg:[14,17,20],last:6,latent:11,later:[6,20],latest:20,latest_model:10,latter:[6,11],layer:[6,10,11,14],lead:5,learn:20,learning_r:10,least:10,left:10,lendth:10,length:[13,17],lesion:17,level:[6,20],librari:20,licenc:18,licens:20,lifecycl:20,like:[17,20],limit:18,line:20,linear:[11,15],linear_sum_assign:15,link:[18,20],linux:20,list:[2,5,6,9,10,13,14,15,17],lmr:14,lmr_filter_typ:13,lmr_radiu:13,load:[4,5,6,10,12,17,20],load_2d_arrai:6,load_data:4,load_dino_model:17,load_full_memmap:6,load_memmap:[6,10],load_memmap_and_clear_scan:6,load_model:10,load_path:17,load_patient_metadata:6,load_scan:6,load_validation_set:6,local:[5,6,10,13,14,20],local_coord:13,local_region:5,locat:[3,6,10,11,17,20],logic:10,loss:10,low:9,m2r2:20,m:[17,20],machin:20,maco:20,made:[17,20],magnitud:[8,10,12],mai:[11,20],main:[12,16],mainwindow:16,major:[13,20],make:20,mani:[10,20],manifest:20,manor:18,manual:20,map:[9,12],markdown:20,mask:[9,10,11,13],masked_data:20,match:[5,9,12,15,20],match_descriptor:9,match_filter_dist:[12,20],match_indic:15,matrix:[3,9,15],max:[14,17],max_area:17,max_ellipse_contour_centre_dist:17,max_ellipse_long_axi:17,max_posit:17,max_shape_from_image_list:14,max_thresh:17,maximum:[12,13,14,20],maximum_source_point:12,maximum_target_point:12,mean:[10,13,14,17,20],mean_axial_thick:6,meanshift:17,meanshiftwithprob:17,measur:20,measuring_w_blob_detector:20,medfilt2d:17,medfilt_kernel:17,medianblur:10,medic:20,membership:17,memmap:[6,7,10,20],memori:[6,12,20],messag:20,met:6,metadata:[6,20],method:[5,17,20],metric:20,might:20,min_area:17,min_area_ratio:17,min_eccentr:17,min_ellipse_long_axi:17,min_posit:17,minim:17,minimis:15,minimum:[6,11,17,20],minor:20,mit:20,mkdir:20,ml:20,mode:10,model:[0,5,10,17,19],model_out:10,model_train:[0,11,19],modif:20,modifi:20,modul:[0,1,18,19,20],modulelist:11,more:[5,13,20],most:20,move:[5,6,10],mse:10,much:17,multipatientaxialstream:[6,17],multipatientload:[6,10],multipl:[6,20],must:[13,14,17,20],n:[6,13,14,17,20],name:[6,7,17,20],napoleon:20,nation:20,natsort:6,navig:20,nbconvert:20,nbia:20,ndarrai:[3,5,6,7,8,9,10,12,13,14,15,17],ndarray_to_memmap:7,nearbi:17,need:[11,20],network:10,neuron:11,neurons_per_dens:[10,11],next:6,nhsx:[18,20],nn:11,nois:20,non:[5,12,20],non_rigid_align:[0,19],none:[2,5,6,10,12,13,14,17],normal:[10,14],normalis:[6,8],note:20,notebook:20,now:20,np:[2,3,5,6,7,8,9,12,13,15,17],npy:[6,20],num_decoder_conv:[10,11],num_dense_lay:[10,11],num_encoder_conv:[10,11],num_input_filt:11,num_output_filt:11,number:[6,10,11,12,13,14,15,17,20],number_of_scan:5,numer:[6,17],numpi:[6,10],nvidia:20,o:20,object:[2,3,4,5,6,10,16,17],occur:[10,14,15],off:10,offset:[5,17],one:[6,9,10,11,17],onli:[9,17],onto:[12,17],open:[4,20],opengl:3,opportun:20,optim:15,optimis:10,option:[2,6,10,13,17,20],orb:9,order:[6,10,12,15,17,20],org:20,organ:20,orient:[3,5,20],origin:[6,10,13,17],other:[10,17,20],otherwis:[6,10,14,17,20],our:20,out:[10,17,20],out_path:17,outlier:9,output:[10,11,13,14,15,17,20],output_height:11,output_path:20,output_width:11,outsid:[10,17],over:20,overal:[10,11],overlai:[2,3,5,12,13,14],overlaid:8,overlap:10,overlay_warp_on_slic:8,overridden:11,packag:[18,19],pad:14,pad_nd:14,page:[4,18],paint:2,painter:2,painter_el:2,paintev:2,pair:15,param:[7,10],paramet:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,17],part:[6,10,11,20],particular:[6,17,20],pass:[4,9,10,11],past:20,patch:[10,11,20],patch_siz:17,path:[4,6,7,10,12,17,20],pathlib:[4,6,7,10,17],patient:[4,6,10,17,20],patient_1:20,patient_2:20,patient_:20,patient_index:10,patient_indic:10,patient_path:6,patientload:6,per:[6,17],perform:[5,9,11,12,14,17,20],phase:[5,13,20],phase_correl:[0,19],phase_correlation_image_process:[0,19],phase_correlation_shift:5,pick:20,pickl:17,pilot:20,pip:20,pipelin:[5,12,18],pixel:[6,7,8,10,13,14,17,20],pixel_arrai:7,pixmap:2,pkl:[17,20],plane:[3,6,10,11],plane_index:10,plane_indic:10,plane_mask_build:10,plot:[3,10],point:[2,9,10,12,13,14,15,17,20],point_match:[0,19],point_threshold:12,pointcloud:3,points_0:15,points_1:15,points_2:15,polici:20,popul:[2,20],portal:20,posit:[6,7,13,15,17,20],possibl:10,pre:[13,17,20],pre_process_data:3,precomput:20,predict:[10,11,17,20],predict_ful:17,predict_proba:17,predict_proba_secondari:17,predict_secondari:17,prefix:[6,7],prepar:20,preprocess:3,present:[4,5,20],press:5,pretrain:10,pretrained_weight:17,previou:17,primari:17,primary_label:17,prior:[13,14],probabilities_im:17,probabl:17,problem:20,process:[8,10,13,20],process_full_scan:[10,20],produc:[17,20],product:20,programat:5,programm:20,progress:20,project:6,proof:20,properti:17,propos:20,protect:18,proteom:20,provid:[6,9,12,14,20],pth:10,publicli:20,pure:6,purpos:18,py:20,pydicom:7,pyqtgraph:3,pyqttablewidget:5,pyside2:[2,3,4,5,16],pytest:20,python:20,pytorch:20,qevent:2,qlabel:2,qmainwindow:16,qmdiarea:4,qmetaobject:[2,3,4,5,16],qpainter:2,qpixmap:2,qt:2,qtcore:[2,3,4,5,16],qtwidget:[2,3,4,5,16],qualifi:20,qwidget:[3,5],r:14,radiolog:20,radiologist:20,radiu:[13,14],random:[6,10,17,20],random_axial_slic:10,random_coronal_slic:10,random_sagittal_slic:10,randomli:[6,17],rang:20,rapidli:20,rather:17,ratio:9,raw:6,raw_transverse_pixel_spacing_and_shap:6,read:12,read_transform:12,real:[10,20],recip:11,recommend:[13,20],record:20,recov:15,rectangl:10,reduc:[12,13],refactor:17,refer:[5,9,12,20],region:[5,6,10,13,20],region_width:13,regist:11,registr:20,regul:20,reject:10,rel:13,releas:20,relev:[5,6,17,20],reload:[6,7,10,17],remain:20,remaind:10,remov:[5,13,14,17,20],renam:20,render:2,replac:17,report:20,repositori:[17,20],repres:[3,8,10,12],reproduct:10,requir:[12,17,20],require_above_thresh:10,rescal:6,rescale_depth:6,rescale_on_load:6,research:[18,20],reset_indic:6,respons:20,restrict:20,result:[5,9,12],retain:9,retriev:20,return_sect:17,return_sub_structur:17,return_v:15,review:20,reweight:10,rgba:3,rich:17,rich_ellips:17,rigid:[5,12,20],roke:18,root_dir:6,roughli:20,round:17,row:[13,14],rule:17,run:[10,11,17,20],s:[9,10,17,20],sa:10,sagitt:[5,10,17],sagittal_width:10,same:[6,8,10,11,13,14,17,20],sampl:17,samples_per_imag:17,satisfactori:20,save:[6,7,10,17,20],save_dir:10,save_freq:10,save_model:10,save_path:10,scale:[2,20],scale_factor:2,scan:[3,5,6,8,10,12,17,18],scan_1:[3,5,6],scan_1_ellips:5,scan_2:[3,5,6],scan_2_ellips:5,scan_index:10,scan_label:5,scan_num_indic:10,scan_tool:[0,19],scan_view:[0,1],scanload:6,scanner:20,scanview:3,scipi:[15,17],screenshot:20,script:20,search:[18,20],second:[5,6,9,10,17,20],second_level_cluster:17,second_try_with_blur:20,second_try_with_blur_displai:20,secondari:17,section:[0,3,5,19],sectioner_kwarg:17,sectioning_model:5,sectioning_out:20,see:[10,17,20],seek:10,seem:10,seen:17,segment:20,select:[4,5,6,17],self:[6,10,13,17,20],send:6,separ:[9,17,20],sequenti:[6,10],serialis:[12,20],set:[5,6,9,10,13,15,17,20],set_data:[3,5],set_imag:2,setup:20,setup_orient:5,sever:20,sh:20,shape:[6,7,8,10,11,12,13,14,15,17],shape_el:6,share:[6,17,20],shell:17,shift:[5,13,20],shift_image_2d:13,shift_nd:13,shift_via_phase_correlation_nd:13,shifts_via_local_region:13,should:[8,9,10,11,13,17,20],show:[3,5,8,12,20],show_navigation_slic:3,show_outlin:10,show_phase_correl:5,shown:20,side:13,siemen:20,sieve_matches_low:9,sift:9,signal:17,silent:11,similar:[13,17,20],similaritytransform:13,simpl:17,simpli:17,sinc:11,singl:[3,5,6,10,13,17,20],single_image_texton_descriptor:17,singledecoderlay:11,singleencoderlay:11,size:[5,10,14,20],skimag:13,skip:17,sklearn:[5,12,17],slice:[3,5,6,7,8,10,11,13,17,20],slice_index:[3,5],slice_loc:7,slice_through_scan:3,slicedirect:5,small:10,smooth:[10,13,14],so:[10,17],softwar:20,solid:14,sourc:[9,12,20],source_point:20,space:[5,6,11,13,17],spatial:15,special:20,specif:[3,5,20],specifi:[3,13],speed:20,sphere:14,sphinx:20,sphinxcontrib:20,sprint:20,squar:10,src:10,stack:[6,10,11],stage:20,stakehold:20,start:[17,18],state:[6,10,20],staticmetaobject:[2,3,4,5,16],step:[6,13,17,20],still:20,stitch:[10,20],store:[5,6,7,10,17],str:[2,5,6,7,9,10,12],strand:20,stream:6,stream_next:6,streamer:6,string:[2,5],structur:[3,17],studi:20,style:[11,17],sub:[13,17],sub_cluster_label:17,sub_structure_class_label:17,subclass:11,subject:20,submodul:19,subpackag:19,subset:17,subtract:10,subwindow:4,success:20,suitabl:[2,15,20],sum:15,supervis:20,support:[10,20],swap:6,symmetr:13,system:20,t:6,tabl:5,table_index:5,take:[10,11,12,13,14],taken:[10,12,20],tar:20,target_point:20,task:20,tcia:20,teacher:17,team:20,technic:20,techniqu:20,tensor:[10,11],term:20,termin:20,test:9,text:[2,20],texton:[17,20],texton_sample_set:17,textonsection:[5,17],texttt:10,than:[14,17,20],thei:17,them:[11,13,14,17],theorem:14,therefor:[17,20],thi:[4,5,6,11,13,16,17,18,20],thick:[6,12,20],thorax:10,those:17,though:20,thre:6,thread:20,three:20,thresh:14,threshold:[6,10,12,17,20],through:[6,10,17],time:[10,20],timeout:20,tissu:[5,17],togeth:[10,17,20],toggl:5,toggle_ellipse_t:5,toggle_ellipse_table_button:5,toggle_non_rigid_2d_control:5,toggle_phase_correlation_control:5,toggle_tissue_sect:5,toggle_warp_displai:5,tool:[4,12],top:[6,10,20],torch:[10,11],total:[6,17],total_sampl:17,touch:20,toward:18,trail:5,train:[5,6,10,11,17,20],train_cluster:17,train_for_iter:10,train_section:20,train_step:10,trainer:20,training_arg:20,transform:[5,9,12,13],transform_3d_volum:12,transform_3d_volume_in_chunk:12,transform_funct:12,translat:13,transvers:6,treat:17,treatment:20,trial:20,trigger:5,trl:20,trust:20,tumor:20,tupl:[3,5,6,12,13,14,15,17],turn:10,two:[5,6,9,10,13,14,15,20],type:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],typic:[13,14,17],ucec:20,ui:[3,4,5,16],ui_stack_window:[0,1],uint16:7,uint8:17,uistackwindow:4,uk:20,under:[18,20],underli:17,unet:11,unit:20,unlabel:20,unless:20,unport:20,unsupervis:20,until:[6,10],up:[10,12,14,17,20],updat:[3,5,20],update_ellipse_t:5,update_local_region:5,update_scan:5,update_scan_toggl:5,upon:[6,20],us:[3,5,6,9,10,11,12,13,14,17,20],usag:20,user:20,uterin:20,util:20,utilis:2,v1:20,valid:[6,17,20],valu:[2,3,5,6,8,10,12,13,14,17],vector:[10,13],version:[6,10,20],via:[13,14,15,20],viabl:20,video:20,view:[4,5,6,10,13,17,20],viewer:4,viewer_data:3,viewer_widget:[0,1],viewerwidget:5,virtual:20,visibl:5,vision:20,visual:20,visualize_attent:17,vit_smal:20,vit_tini:17,volum:[3,10,12,17,20],volume_plot:3,volumetr:12,voxel:12,vram:10,w_pretrained_dino:20,wa:[6,10,17,20],want:[7,17],warp:[5,8,12],warp_slic:8,we:20,web:20,week:20,weight:[10,13],well:[10,17,20],were:[10,17,20],when:[2,10,17,20],where:[6,7,13,14,17,20],wherev:17,whether:[6,10,13,14,17],which:[5,6,10,13,14,17,20],whilst:20,white:20,wider:20,widget:[3,4,5,16],width:10,window:[2,16,20],wise:[6,17],wish:[7,17],within:[2,4,6,10,11,17,20],without:[6,7,20],work:[10,11,18],would:20,write:[7,12,20],write_transform:12,x:[2,3,5,11],xyz:10,y:[2,3,5],ye:20,yet:6,you:[6,7,17,20],your:20,z:[3,5,6],zero:[6,10,13,14],zero_cross:14,zero_crossings_thresh:13,zeroth:[13,15]},titles:["ai_ct_scans package","ai_ct_scans.GUI package","ai_ct_scans.GUI.image_viewer module","ai_ct_scans.GUI.scan_viewer module","ai_ct_scans.GUI.ui_stack_window module","ai_ct_scans.GUI.viewer_widget module","ai_ct_scans.data_loading module","ai_ct_scans.data_writing module","ai_ct_scans.image_processing_utils module","ai_ct_scans.keypoint_alignment module","ai_ct_scans.model_trainers module","ai_ct_scans.models module","ai_ct_scans.non_rigid_alignment module","ai_ct_scans.phase_correlation module","ai_ct_scans.phase_correlation_image_processing module","ai_ct_scans.point_matching module","ai_ct_scans.scan_tool module","ai_ct_scans.sectioning module","NHS AI Lab Skunkworks project: CT Alignment and Lesion Detection","ai_ct_scans","NHS AI Lab Skunkworks project: CT Alignment and Lesion Detection"],titleterms:{ai:[18,20],ai_ct_scan:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19],align:[18,20],background:20,content:18,cpd:20,ct:[18,20],data:20,data_load:6,data_writ:7,detect:[18,20],develop:20,dino:20,document:20,execut:20,experi:20,gener:20,get:20,gui:[1,2,3,4,5,20],image_processing_util:8,image_view:2,infil:20,instal:20,intend:20,keypoint_align:9,known:20,lab:[18,20],lesion:[18,20],licenc:20,limit:20,mask:20,model:[11,20],model_train:10,modul:[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],nh:[18,20],non_rigid_align:12,packag:[0,1],phase_correl:13,phase_correlation_image_process:14,pipelin:20,point_match:15,project:[18,20],protect:20,purpos:20,sampl:20,scan:20,scan_tool:16,scan_view:3,section:[17,20],select:20,skunkwork:[18,20],start:20,structur:20,submodul:[0,1],subpackag:0,test:20,tissu:20,tool:20,transform:20,ui_stack_window:4,viewer_widget:5}})