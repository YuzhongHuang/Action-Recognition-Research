-- George Chen hc25@rice.edu
-- Convert the video and its labels to Tensor

require 'image'
require 'ffmpeg'
require 'xlua'

-- Parse command-line arguments

op = xlua.OptionParser('extract_video_frames.lua [options]')
op:option{'-d', '--dir', action='store', dest='dir', help='directory to load', default = 'KTH', req=true}
op:option{'-e', '--ext', action='store', dest='ext', help='only load files of this extension', default='avi'}
op:option{'-f', '--fps', action='store', dest='fps', help='desired fps of the videos', default='10'}
op:option{'-l', '--length', action='store', dest='length', help='desired length of the videos', default='10'}
op:option{'-w', '--width', action='store', dest='width', help='desired width of the videos', default='70'}
op:option{'-h', '--height', action='store', dest='height', help='desired height of the videos', default='40'}
op:option{'-c', '--encoding', action='store', dest='encoding', help='image encoding type', default = 'jpg'}
op:option{'-v', '--verbose', action='store', dest='verbose', help='verbose', default = true}

opt = op:parse()
op:summarize()

-- Define ls function
function ls(path) return sys.split(sys.ls(path),'\n') end 

vid_nframes = opt.fps * opt.length-20; 
	-- have some slack on the number of frames, as ffmpeg sometimes is not accurate with timestamp
vid_path = opt.dir
vid_subpath = ls(vid_path)
frames = torch.zeros(1,vid_nframes,opt.height,opt.width);
labels = torch.zeros(1);
classes = vid_subpath;

-- Extract frames, putting them into a tensor
for i=1, #vid_subpath do 
	print('i='..i)
	vid_files = ls(vid_path ..'/'.. vid_subpath[i])
	for j = 1, #vid_files do
		--print('j='..j)
		current_vid_file = vid_path ..'/'.. vid_subpath[i] .. '/' .. vid_files[j]		
		local source = ffmpeg.Video{path=current_vid_file, encoding=opt.encoding, 
		fps=opt.fps, lenght=opt.lenght, width=opt.width, height=opt.height, delete=false, load=true, silent=opt.verbose}
		local rawFrames = source:totensor{}
		local current_frame = rawFrames[{ {1, vid_nframes}, {1}, {}, {}  }]:transpose(1,2); -- from 100x3x40x70 to 1x80x40x70
		frames = frames:cat(current_frame,1);
		local current_label = torch.Tensor({i});
		labels = labels:cat(current_label,1);		
	end
	
end

labels = labels[{{2,-1}}];
frames = frames[{{2,-1},{},{},{}}];

datasets = {
   vids = frames,
   labels = labels,
   classes = classes
}

torch.save('datasets.t7', datasets);