%% create files regarding subjects family information used in TOPF

clc;
clear;

T = readtable('/RESTRICTED_7T.csv'); % restricted info of subjects, available from http://db.humanconnectome.org/
subid = importdata('./subid.txt'); % subjects id list to be investigated

for i = 1:length(subid)
    ind = find(T.Subject == subid(i));
    familyid(i,1) = T.Family_ID(ind);
end

[nMem,FamID] = groupcounts(familyid); % unique Family ID

nFam=length(FamID);
for i=1:nFam
    FamSubID{i} = find(contains(familyid,FamID{i}));
end

%% create file: family ID for python, sub index -1
for i=1:length(FamID)
temp{i} =(FamSubID{i}-1)'; % index from 0
end
temp = temp';
v = [FamID  temp];
varname={'FamilyID','FamilyMembers'};
T = array2table(v,'VariableNames',varname);
fname ='./FamilyIDinfo.csv';
writetable(T,fname, 'Delimiter',',')

fid = fopen('./famid.txt','w');
CT = FamID.';
fprintf(fid,'%s\n', CT{:});
fclose(fid)


%% create file for python, sub index -1
for i=1:length(subid)
temp{i} =(find(contains(familyid,familyid{i}))-1)';
end
temp = temp';
x=[1:length(subid)]'-1;
x=subid;
v = [num2cell(x)  temp];
varname={'Subject','FamilyMembers'};
T = array2table(v,'VariableNames',varname);
fname ='./Familyinfo.csv';
writetable(T,fname, 'Delimiter',',')

