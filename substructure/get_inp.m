%%
bain_ebsd = EBSD.load('S1100Q_pos1_Mod.ang');
bain_ebsd = bain_ebsd('Iron (Alpha)');
mtexdata martensite;
%%
[grains,bain_ebsd.grainId] = calcGrains(bain_ebsd('indexed'), 'angle', 3*degree);

% remove small grains
bain_ebsd(grains(grains.grainSize < 3)) = [];

% reidentify grains with small grains removed:
[grains,bain_ebsd.grainId] = calcGrains(bain_ebsd('indexed'),'angle',3*degree);
grains = smooth(grains,5);
%%
csBCC = bain_ebsd.CSList{2};
csFCC = ebsd.CSList{3};
%%
job = parentGrainReconstructor(bain_ebsd,grains);
job.p2c = orientation.KurdjumovSachs(csFCC, csBCC);

% define child orientation with mean orientation of neighbouring grains
grainPairs = grains.neighbors;

% compute an optimal parent to child orientation relationship
job.calcParent2Child;
%%
histogram(job.fit./degree, 50);
xlabel('disorientation angle');
%%
% compute the misfit for all child to child grain neighbours
[fit,c2cPairs] = job.calcGBFit;

% select grain boundary segments by grain ids
[gB,pairId] = job.grains.boundary.selectByGrainId(c2cPairs);

% plot the child phase
plot(bain_ebsd('Iron (Alpha)'),bain_ebsd('Iron (Alpha)').orientations,'figSize','large','faceAlpha',0.5)

% and on top of it the boundaries colorized by the misfit
hold on;
% scale fit between 0 and 1 - required for edgeAlpha
plot(gB, 'edgeAlpha', (fit(pairId) ./ degree - 2.5)./2 ,'linewidth',2);
hold off
%%
job.calcGraph('threshold',2.5*degree,'tolerance',2.5*degree);
job.clusterGraph('inflationPower',1.6)
%%
job.calcParentFromGraph
plot(job.parentGrains,job.parentGrains.meanOrientation)
%%
job.calcVariants

% associate to each packet id a color and plot
color = ind2color(job.transformedGrains.packetId);
plot(job.transformedGrains,color,'faceAlpha',0.5)
hold on
parentGrains = smooth(job.parentGrains,10);
plot(parentGrains.boundary,'linewidth',3);
%%
block_id = grains.id;
pag_id = job.mergeId;
pak_id = job.packetId;
phi1 = grains.meanOrientation.phi1./degree;
PHI = grains.meanOrientation.Phi./degree;
phi2 = grains.meanOrientation.phi2./degree;

%%
[omega_pag,a,b] = fitEllipse(parentGrains);
[omega_block,w,block_thickness] = fitEllipse(grains);
%%
pag_phi1 = parentGrains.meanOrientation.phi1./degree;
pag_Phi = parentGrains.meanOrientation.Phi./degree;
pag_phi2 = parentGrains.meanOrientation.phi2./degree;
grain_id = parentGrains.id;
writematrix([a,b,omega_pag,pag_phi1,pag_Phi,pag_phi2,grain_id],'pag_info.csv')
writematrix([block_id,pag_id,pak_id,phi1,PHI,phi2,block_thickness],'bainite.csv');