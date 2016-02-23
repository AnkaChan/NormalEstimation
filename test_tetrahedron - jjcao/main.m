clear; clc; close all;
%% changing the selection of center %% need do
addpath ('../toolbox/jjcao_io')
addpath ('../toolbox/jjcao_point')
addpath ('../toolbox/jjcao_common')
addpath ('../toolbox/jjcao_interact')
addpath ('../toolbox/kdtree')
addpath ('../toolbox/jjcao_plot')

addpath('../toolbox/zj_fitting')
addpath('../toolbox/zj_deviation')

addpath('../toolbox/cvx')
cvx_setup
%% debug options
ADDNOISE = 0;
TP.debug_data = 1;
TP.debug_taproot = 0 ;

%% algorithm options
TP.k_knn_feature = 70; % tunable arguments
TP.k_knn_normals = 30;
TP.k_knn  = 120;

TP.sigma_threshold = 0.05;
TP.ran_num = 100 ;
%% read input && add noise && plot it && build kdtree
% [~ , faces] = read_off('..\..\data\fandisk_103K.off');
% orientation_all  = compute_normal_mesh(P.pts' , faces') ;
% orientation_all = orientation_all' ;
% outfile = '..\..\data\fandisk_103K1_test1.off' ;
[P.pts orientation_all] = read_noff('..\..\data\tetrahedron_5K5_noise04.off');
outfile = '..\..\data\tetrahedron_5K5_noise04_test5.off' ;

point_number = size(P.pts , 1) ;
bbox = [min(P.pts(:,1)), min(P.pts(:,2)), min(P.pts(:,3)), max(P.pts(:,1)), max(P.pts(:,2)), max(P.pts(:,3))];
bx = bbox(4)-bbox(1);by = bbox(5)-bbox(2);bz = bbox(6)-bbox(3);
rs = bbox(4:6)-bbox(1:3);
diameter = sqrt(dot(rs,rs));

% add noise
if ADDNOISE
    type = 'gaussian';%type = 'random';% type = 'gaussian';% type = 'salt & pepper';
    base = 'average_edge';%base = 'average_edge'% base = 'diagonal_line';
    p3 = 0.01;
    kdtree = kdtree_build(P.pts);
    pts = pcd_noise_point(P.pts, type, base, p3,kdtree);
    kdtree_delete(kdtree);
end

% build kdtree
P.kdtree = kdtree_build(P.pts);
%% show the density
[l density] = compute_average_radius(P.pts,30,P.kdtree) ;
area_w = density .^ 2 ;
% figure('Name','mesh_show_disity'); movegui('southeast'); set(gcf,'color','white');
% options.face_vertex_color = desity' ;
% options.edge_color = 1;
% h = plot_mesh(P.pts, faces, options);view3d rot;
% colormap(jet); lighting none;

% figure('Name','points_show_disity'); set(gcf,'color','white');set(gcf,'Renderer','OpenGL');
% movegui('northeast');
% md = mean(desity);
% for j = 1 : 10 : size(P.pts,1)
%     color_density = [desity(j) 0 md]';
%     plot3(P.pts(j , 1) , P.pts(j,2) , P.pts(j,3) , '.' , 'MarkerSize' , 20 , 'color' , color_density ) ; hold on ;
% end
% axis off;axis equal;
% view3d rot;
% hold on
%% compute initial features (a ribbon)
[sigms , normVectors , errs , normals_comW] = compute_points_sigms_normals_two(P.pts, TP.k_knn_feature, P.kdtree, TP.k_knn_normals);
TP.feature_threshold = feature_threshold_selection(sigms,TP);
TP.id_feature = find(sigms > TP.feature_threshold);

fitting_threshold = fitting_threshold_selection(errs,TP.id_feature);
inner_threshold = 1.5 * fitting_threshold;

%%disp('initial features: ');
% if TP.debug_data;
%     figure('Name','Input'); set(gcf,'color','white');set(gcf,'Renderer','OpenGL');
%     movegui('northeast');
%     scatter3(P.pts(:,1),P.pts(:,2),P.pts(:,3),10,'.','MarkerEdgeColor',GS.CLASS_COLOR5);  hold on;
%     scatter3(P.pts(TP.id_feature,1),P.pts(TP.id_feature,2),P.pts(TP.id_feature,3),30,'.','MarkerEdgeColor',GS.CLASS_COLOR1);  hold on;
%     axis off;axis equal;
%     view3d rot; % vidw3d zoom; % r for rot; z for zoom;
% end

nFeature = length(TP.id_feature)
nSample = size(P.pts,1);
noncompute_count = 0;
tic
%% construct neighboor matrix
feature_normal_r = 0.042 * diameter ;
neigh_matrix = cell(1 , nSample) ;
neigh_matrix_indicate = sparse(nSample , nSample) ;
for  i = 1 : nSample
    idxs = kdtree_k_nearest_neighbors(P.kdtree,P.pts(i,:),30);
    neigh_matrix{i} = idxs ;
    neigh_matrix_indicate(i , neigh_matrix{i}) = 1 ;
    
%         [idxs, ~] = kdtree_ball_query(P.kdtree,P.pts(i,:)',feature_normal_r);
%         neigh_matrix{i} = idxs ;
%         neigh_matrix_indicate(i , neigh_matrix{i}) = 1 ;
end
%% compute nomals
load('EW_function.mat')
for i = 1 : nFeature
    %
    ii =  TP.id_feature(i);
    
%     figure('Name','Input'); set(gcf,'color','white');set(gcf,'Renderer','OpenGL');
%     movegui('northeast');
%     scatter3(P.pts(:,1),P.pts(:,2),P.pts(:,3),10,'.','MarkerEdgeColor',GS.CLASS_COLOR5);  hold on;
%     scatter3(P.pts(neigh_matrix{ii},1),P.pts(neigh_matrix{ii},2),P.pts(neigh_matrix{ii},3),30,'.','MarkerEdgeColor',GS.CLASS_COLOR1);  
%     axis off;axis equal;
%     view3d rot; % vidw3d zoom; % r for rot; z for zoom;
    
    can_nei_id = find(neigh_matrix_indicate(: , ii) ~= 0) ;
    
    can_nei = neigh_matrix(can_nei_id) ;
    orientation = orientation_all(ii,:) ;
    
    mp = mean(P.pts(neigh_matrix{ii}(1:7),:));
    [normVectors(ii,:) noncompute]= compute_normal_jjcao_EACH1(P.pts , can_nei , orientation ,area_w , mp) ;
    
    noncompute_count = noncompute_count + noncompute ;
%     if TP.debug_data;
%         if normVectors(ii,:) * orientation' < cos(10*pi/180)
%         figure('Name','Input'); set(gcf,'color','white');set(gcf,'Renderer','OpenGL');
%         movegui('northeast');
%         scatter3(P.pts(:,1),P.pts(:,2),P.pts(:,3),10,'.','MarkerEdgeColor',GS.CLASS_COLOR5);  hold on;
%         %scatter3(P.pts(knn,1),P.pts(knn,2),P.pts(knn,3),30,'.','MarkerEdgeColor',GS.CLASS_COLOR1);  hold on;
%         x = [P.pts(ii,:) ; P.pts(ii,:) + normVectors(ii,:)] ;
%         plot3(x(:, 1),x(:,2),x(:,3) , 'LineWidth' , 2) ; hold on
%         y = [P.pts(ii,:) ; P.pts(ii,:) + orientation] ;
%         plot3(y(:, 1),y(:,2),y(:,3) , 'r' , 'LineWidth' , 2) ; hold on
%         axis off;axis equal;
%         view3d rot; % vidw3d zoom; % r for rot; z for zoom;
%         close all
%         end
%     end
            
end
toc
%%
kdtree_delete(P.kdtree);
write_noff(outfile ,P.pts,  normVectors)

figure('Name','Input'); set(gcf,'color','white');set(gcf,'Renderer','OpenGL');
movegui('northeast');
scatter3(P.pts(:,1),P.pts(:,2),P.pts(:,3),30,'.','MarkerEdgeColor',GS.CLASS_COLOR5);  hold on;
for i = 1 : nFeature
    %
    ii = TP.id_feature(i);
    x = [P.pts(ii,:) ; P.pts(ii,:) + 1/4*orientation_all(ii,:)] ;
    plot3(x(:, 1),x(:,2),x(:,3) , 'LineWidth' , 1) ; hold on
    %             y = [P.pts(ii,:) ; P.pts(ii,:) + orientation] ;
    %             plot3(y(:, 1),y(:,2),y(:,3) , 'r' , 'LineWidth' , 2) ; hold on
end
axis off;axis equal;
view3d rot; % vidw3d zoom; % r for rot; z for zoom;

load('tetrahedron_5K5_noise05_truenormal.mat')
deviation_vector = compute_deviation_vector_feature(true_normals, normVectors,10);
bad_id = find(deviation_vector > (pi/180)*10) ;
good_id = setdiff(1:nSample , bad_id) ;
figure('Name','Input'); set(gcf,'color','white');set(gcf,'Renderer','OpenGL');
movegui('northeast');
scatter3(P.pts(good_id,1),P.pts(good_id,2),P.pts(good_id,3),300,'.','MarkerEdgeColor',GS.CLASS_COLOR4);  hold on;
scatter3(P.pts(bad_id,1),P.pts(bad_id,2),P.pts(bad_id,3),300,'.','MarkerEdgeColor',GS.CLASS_COLOR1);  hold on;
axis off;axis equal;
view3d rot; % vidw3d zoom; % r for rot; z for zoom;

figure('Name','Input'); set(gcf,'color','white');set(gcf,'Renderer','OpenGL');
movegui('northeast');
scatter3(P.pts(good_id,1),P.pts(good_id,2),P.pts(good_id,3),30,'.','MarkerEdgeColor',GS.CLASS_COLOR4);  hold on;
scatter3(P.pts(bad_id,1),P.pts(bad_id,2),P.pts(bad_id,3),100,'.','MarkerEdgeColor',GS.CLASS_COLOR1);  hold on;
for i = 1 : length(bad_id)
    %
    ii = bad_id(i);
    x = [P.pts(ii,:) ; P.pts(ii,:) + 1/4 * normVectors(ii,:)] ;
    plot3(x(:, 1),x(:,2),x(:,3) , 'LineWidth' , 2) ; hold on
    y = [P.pts(ii,:) ; P.pts(ii,:) + 1/4 * orientation_all(ii,:)] ;
    plot3(y(:, 1),y(:,2),y(:,3) , 'r' , 'LineWidth' , 2) ; hold on
end
axis off;axis equal;
view3d rot; % vidw3d zoom; % r for rot; z for zoom;