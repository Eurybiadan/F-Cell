% Rob Cooper 06-30-2017
%
% This script is an implementation of the algorithm outlined by Salmon et
% al 2017: "An Automated Reference Frame Selection (ARFS) Algorithm for 
% Cone Imaging with Adaptive Optics Scanning Light Ophthalmoscopy"
%

clear;
close all;

STRIP_SIZE = 40;

locInd=1; % Location index

% For Debug
mov_path = {pwd,...
            pwd,...
            pwd};
stack_fname = {'NC_11049_20170629_confocal_OD_0000_desinusoided.avi',...
               'NC_11049_20170629_avg_OD_0000_desinusoided.avi',...
               'NC_11049_20170629_split_det_OD_0000_desinusoided.avi'};

for modalityInd=1 : 1%length(stack_fname) 
    vidReader = VideoReader( fullfile(mov_path{locInd,modalityInd}, stack_fname{locInd, modalityInd}) );
    
    i=1;
    while(hasFrame(vidReader))
        image_stack(:,:,i,modalityInd) = uint8(readFrame(vidReader));        
        frame_mean(i,modalityInd) = mean2(image_stack(:,:,i,modalityInd));
        i = i+1;
    end
    numFrames = i-1;
    
    % Get some basic heuristics from each modality.
    mode_mean(modalityInd) = mean(frame_mean(:,modalityInd));
    mode_dev(modalityInd) = std(frame_mean(:,modalityInd));

    frame_contenders(:,modalityInd) = (1:numFrames);
    

    strip_inds = 0:STRIP_SIZE:size(image_stack(:,:,1, 1),2);
    strip_inds(1) = 1;
    if strip_inds(end) ~= size(image_stack(1:40,:,93,modalityInd),2)
        strip_inds = [strip_inds size(image_stack(:,:,1,modalityInd),2)];
    end
    num_strips = length(strip_inds)-1;
    
    mean_contenders = false(1,numFrames);
    for f=1:numFrames        
        mean_contenders(f) =  (frame_mean(f,modalityInd) < mode_mean(modalityInd)+2*mode_dev(modalityInd)) &&...
                              (frame_mean(f,modalityInd) > mode_mean(modalityInd)-2*mode_dev(modalityInd));
    end
    
    frame_contenders = frame_contenders(mean_contenders,modalityInd);
    
    numFrames = length(frame_contenders);
    
    radon_bandwidth = zeros(numFrames,num_strips);
    %%
    tic;
    if exist('parfor','builtin') == 5 % If we can multithread it, do it!
        parfor f=1:numFrames
            
            frame_ind=frame_contenders(f,modalityInd);
            
            for s=1:num_strips
    
                % Get the log power spectrum for us to play with
                pwr_spect = ( abs(fftshift(fft2(image_stack(strip_inds(s):strip_inds(s+1),:, frame_ind),512, 512))).^2);
                % From our padding, the center vertical frequency will be
                % garbage- remove it for our purposes.
                pwr_spect = log10(pwr_spect(:,[1:256 258:512]));
                
                % Threshold is set using the upper 2 std devs
                thresh_pwr_spect = ( pwr_spect>(mean(pwr_spect(:))+2*std(pwr_spect(:))) );
                
                radoned = radon( thresh_pwr_spect );                

                % Determine the minimum and maximum FWHM
%                 tic;
                halfmax = repmat(max(radoned)./2,[727 1]);
%                 toc;
                fwhm = sum(radoned>halfmax);
                
                % 
                radon_bandwidth(f,s) = max(fwhm)-min(fwhm);

            end
        end        
        
        threshold = mean(radon_bandwidth(:))+ 2*std(radon_bandwidth(:));
        % After thresholding and removal, update the contenders list.
        frame_contenders = frame_contenders(~any(radon_bandwidth > threshold,2));
        
        
        frm1 = double(image_stack(:,:, frame_contenders(1, modalityInd), modalityInd));
        [m, n] = size(frm1);
        paddiffm = (size(frm1,1)*2)-1-m;
        paddiffn = (size(frm1,2)*2)-1-n;
        
%         fft_frm = zeros((size(frm1,1)*2)-1,(size(frm1,2)*2)-1, length(frame_contenders));
%         
%         parfor f=1:numFrames
%             frame_ind = frame_contenders(f, modalityInd);
%             frm = double(image_stack(:,:,frame_ind, modalityInd));
%             
%             padfrm = padarray(frm,[paddiffm paddiffn],0,'post');
%             fft_frm(:,:,f) =  fft2( padfrm );
%         end
        
%         frm1 = double(image_stack(:,:, frame_contenders(1, modalityInd), modalityInd));
        
        % Make a mask to remove any edge effects.
        [maskdistx, maskdisty] = meshgrid( 1:(size(frm1,2)*2)-1, 1:(size(frm1,1)*2)-1);
        
        maskdistx = maskdistx-(size(maskdistx,2)/2);
        maskdisty = maskdisty-(size(maskdistx,1)/2);
        
        xcorr_mask = sqrt(maskdistx.^2 + maskdisty.^2) <400;
        % Determine the number of pixels that will overlap between the two
        % images at any given point.
        numberOfOverlapPixels = arfs_local_sum(ones(size(frm1)),size(frm1,1),size(frm1,2));
        
        [m, n] = size(frm1);
        paddiffm = (size(frm1,1)*2)-1-m;
        paddiffn = (size(frm1,2)*2)-1-n;
        
        padfrm1 = padarray(frm1,[paddiffm paddiffn],0,'post');
        
        fft_frm1 =  fft2( padfrm1 );
        
        ncc = zeros(length(frame_contenders)-1,1);
        ncc_offset = zeros(length(frame_contenders)-1,2);
        
        for f=2:length(frame_contenders)
            
            frame_ind2 = frame_contenders(f, modalityInd);
            frm2 = double(image_stack(:,:,frame_ind2, modalityInd));
                         
            padfrm2 = padarray(frm2,[paddiffm paddiffn],0,'post'); 
            
            % Denominator for NCC (Using the local sum method used by
            % MATLAB and J.P Lewis.            
            local_sum_F = arfs_local_sum(frm1,m,n);
            local_sum_F2 = arfs_local_sum(frm1.*frm1,m,n);
            
            rotfrm2 = rot90(frm2,2); % Need to rotate because we'll be conjugate in the fft (double-flipped)
            local_sum_T = arfs_local_sum(rotfrm2,m,n);
            local_sum_T2 = arfs_local_sum(rotfrm2.*rotfrm2,m,n);

            % f_uv, assumes that the two images are the same size (they should
            % be for this application)- otherwise we'd need to calculate
            % overlap, ala normxcorr2_general by Dirk Padfield
            denom_F = max( local_sum_F2- ((local_sum_F.^2)./numberOfOverlapPixels), 0);
            % t
            denom_T = max( local_sum_T2- ((local_sum_T.^2)./numberOfOverlapPixels), 0);

            fft_frm2 = fft2( padfrm2 );
            
            % Numerator for NCC
            numerator = fftshift( real(ifft2( fft_frm1 .* conj( fft_frm2 ) ) ));
            
%             pcorr= ifft2( (fft_frm1 .* conj( fft_frm2 ))./abs(fft_frm1 .* conj( fft_frm2 )) );
            
            numerator = numerator - local_sum_F.*local_sum_T./numberOfOverlapPixels;
            
            denom = sqrt(denom_F.*denom_T);
            
            ncc_frm = (numerator./denom);
            
            masked_ncc_frm = (ncc_frm.*xcorr_mask);
            masked_ncc_frm = masked_ncc_frm +( (~xcorr_mask).* min(masked_ncc_frm(:)) );
            
            [ncc(f-1), ncc_ind_offset]  = max(masked_ncc_frm(:));
            
            [xoff, yoff]= ind2sub(size(masked_ncc_frm),ncc_ind_offset);                        
            ncc_offset(f-1,:) = [xoff yoff];
            
%             if f==65
%                 figure(1);imshowpair(frm1,frm2,'montage')
%                 mean2(frm2)
%                 figure(2);imagesc(masked_ncc_frm); axis image;
%                 [thexcorr,nft]=normxcorr2_general(frm2,frm1);
%                 figure;imagesc(ncc_frm-thexcorr);
%             end
            
            % Frame 2 is now frame 1
            fft_frm1 = fft_frm2;
            frm1 = frm2;
            
        end
        
        ncc_offset(:,1)=ncc_offset(:,1)-size(image_stack(:,:,1),1);
        ncc_offset(:,2)=ncc_offset(:,2)-size(image_stack(:,:,1),2);

        figure; plot(cumsum(ncc_offset(:,1)),cumsum(ncc_offset(:,2)),'.');
        
        
        
        
        
        for f=1:length(frame_contenders)-1
            frame_ind=frame_contenders(f,modalityInd);
            frm1 = double(image_stack(:,:,frame_ind,modalityInd));
            frm2 = double(image_stack(:,:,frame_ind+1,modalityInd));
            xcored(f) = std2( ( ((frm2-mean2(frm2))./std2(frm2)) - ((frm1-mean2(frm1))./std2(frm1))) );
        end
        hist(xcored,20);
    else
        
    end
    toc;
end


    


                