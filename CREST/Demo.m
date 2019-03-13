function info = Demo()

varargin=cell(1,2);

varargin(1,1)={'train'};
varargin(1,2)={struct('gpus', 1)};

run ../matconvnet/matlab/vl_setupnn ;
addpath ../matconvnet/examples ;

opts.expDir = 'exp/' ;
opts.dataDir = 'exp/data/' ;
opts.modelType = 'tracking' ;
opts.sourceModelPath = 'exp/models/' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

% experiment setup
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
opts.imdbStatsPath = fullfile(opts.expDir, 'imdbStats.mat') ;
opts.vocEdition = '11' ;
opts.vocAdditionalSegmentations = false ;

global resize;
display=1;

g=gpuDevice(1);
clear g;                             

base_path = '/media/cjh/datasets/tracking/OTB100/';
% choose name of the VOT sequence
sequence_name = choose_video(base_path);

[config]=config_list(base_path,sequence_name);

result=CREST_tracking(opts,varargin,config,display);        
       



