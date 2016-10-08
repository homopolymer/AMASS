#!/usr/bin/env python

import os
import sys
import pysam
import logging
import argparse
import numpy as np
import pandas as pd
import itertools as its
from Bio import Phylo,AlignIO,SeqIO,Seq
from ete3 import Tree,Phyloxml
from glob import glob
from subprocess import call
from scipy.stats import poisson
from collections import defaultdict
from multiprocessing import Pool,cpu_count

phylolevels = ['Kingdom','Phylum','Class','Order','Family','Genus','Species']
# gene identity
gg = ''
# path to gene sequence
ggref = ''
# path to gene taxonomy
ggtax = ''
# path to bowtie2 index
ggidx = ''
# path to vsearch
vsearch = ''
# path to StrainCall
straincall = ''
# cores
cpus = cpu_count()
# threshold for searching candidate clades
rel_abun_thresh = 0.01
# threshold for searching candidate clades
abs_abun_thresh = 10
# threshold for dereplication
derep_minsize = 10
# initial sequencing error
error_rate = 0.01
# threshold for poisson testing
poisson_thresh = 1e-10
# OTU dissimilarity threshold
otu_thresh = 0.97

# current working directory
cwd = os.getcwd()
# temporary directory for bash scripts
tmp_scripts = os.path.join(cwd,'temp_scripts')
call(['mkdir','-p',tmp_scripts])
# temporary directory for alignments
tmp_aligns = os.path.join(cwd,'temp_aligns')
call(['mkdir','-p',tmp_aligns])
# temporary directory for clustering
tmp_clusters = os.path.join(cwd,'temp_clusters')
call(['mkdir','-p',tmp_clusters])

# total abundance of the given data
total_abun = 0

# pvalue threshold for testing whether a rare clade is error or not
rare_test_p = 1e-8

# error amplicon distribution
error_amplicon = pd.DataFrame(columns=['ErrorAmpliconNum','Frequency'])

# verbosity
verbose = False

def parse_parameters(params_file):
    def converter(x):
        x = str(x)
        return x.strip()

    global gg,ggref,ggidx,ggtax,vsearch,straincall,cpus,rel_abun_tresh,\
           abs_abun_thresh,derep_minsize,error_rate,poisson_thresh,otu_thresh,rare_test_p

    params = pd.read_table(params_file,index_col=False,header=None,sep='=',engine='python',
                           names=['Option','Value'],
                           converters={'Option':converter,'Value':converter})
    gg = params.loc[params.Option=='gg','Value'].squeeze()
    ggref = params.loc[params.Option=='ggref','Value'].squeeze()
    ggidx = params.loc[params.Option=='ggidx','Value'].squeeze()
    ggtax = params.loc[params.Option=='ggtax','Value'].squeeze()
    vsearch = params.loc[params.Option=='vsearch','Value'].squeeze()
    straincall = params.loc[params.Option=='straincall','Value'].squeeze()
    cpus = min(cpus,int(params.loc[params.Option=='cpus','Value'].squeeze()))
    rel_abun_thresh = float(params.loc[params.Option=='rel_abun_thresh','Value'].squeeze())
    abs_abun_thresh = int(params.loc[params.Option=='abs_abun_thresh','Value'].squeeze())
    derep_minsize = int(params.loc[params.Option=='derep_minsize','Value'].squeeze())
    error_rate = float(params.loc[params.Option=='error_rate','Value'].squeeze())
    poisson_thresh = float(params.loc[params.Option=='poisson_thresh','Value'].squeeze())
    otu_thresh = float(params.loc[params.Option=='otu_thresh','Value'].squeeze())
    rare_test_p = float(params.loc[params.Option=='rare_test_pvalue','Value'].squeeze())


def exec_script(bash_script):
    call(['bash',bash_script])
    return None


def exec_command(cmd):
    from numpy.random import choice
    rid = ''.join(map(str,choice(10000,10,replace=True)))
    
    script = os.path.join(tmp_scripts,rid+'.sh')
    with open(script,'w') as f:
        f.write(cmd)
    
    exec_script(script)


def load_greengene_tax(gene_tax_file):
    gene_tax = pd.read_table(gene_tax_file,index_col=False,sep='\t|;',engine='python',
                         names=['Gene','Kingdom','Phylum','Class','Order','Family','Genus','Species'],
                         converters = dict([('Gene',str)]+
                                          [(t,lambda x:x.strip()) for t in ['Gene','Kingdom','Phylum','Class',
                                                                            'Order','Family','Genus','Species']]))
    return gene_tax

def load_gene_index(gene_ref_file):
    gene_bed = pd.read_table(gene_ref_file+'.fai',index_col=False,names=['Gene','End'],usecols=[0,1])
    gene_bed['Start'] = [1]*gene_bed.shape[0]
    return gene_bed


def parse_sample_metadata(sample_metadata_file):
    '''Sample metadata format'''
    sample_metadata = pd.read_table(sample_metadata_file,delim_whitespace=True,index_col=False)
    sample_metadata.columns = map(lambda x:x.replace('#',''),sample_metadata.columns)
    return sample_metadata

def parse_sample_read(sample_read_file):
    sample_read = pd.read_table(sample_read_file,index_col=False,header=None,names=['SampleID','Read'])
    return sample_read

def build_taxonomic_tree(gene_tax_abun,stoplevel='genus'):
    '''The input is a table storing both taxonomy and abundance.
       The level parameter specifies the depth of the tree. Allowed
       values of level are kingdom,phylum, class, order, family, genus.'''
    def get_node(nodename):
        if nodename in lookup:
            return lookup[nodename]
        else:
            lookup[nodename] = Tree(name=nodename)
            return lookup[nodename]
        
    # a lookup table to quickly query the tree of a given clade
    lookup = {}
    
    # create a tree with root
    tree = Tree(name='root')
    lookup['root'] = tree
    
    # loop over phylolevels
    stop = False
    for i,t in enumerate(phylolevels):

        if stop:
            break
            
        # current level is kingdom
        if t == 'Kingdom':
            # parent node name
            parent_name = 'root'
            # iterate over taxa
            for child_taxon,child_abun in gene_tax_abun.groupby(t).agg({'Abundance':sum}).iterrows():
                parent = get_node(parent_name)
                # add new node
                child_name = child_taxon
                parent.add_child(get_node(child_name))
                # add node feature
                get_node(child_name).add_features(abundance=child_abun.values[0],taxon=child_taxon)                
        else: # low ranked level
            # iterate over taxa
            for index,subgroup in gene_tax_abun.groupby(phylolevels[:i+1]):
                parent_name = ';'.join(subgroup[phylolevels[:i]].iloc[0,:])
                child_name = ';'.join(subgroup[phylolevels[:i+1]].iloc[0,:])
                child_taxon = subgroup[phylolevels[i]].values[0]
                child_abun = subgroup['Abundance'].sum()
                # add new node
                parent = get_node(parent_name)
                parent.add_child(get_node(child_name))
                # add node feature
                get_node(child_name).add_features(abundance=child_abun,taxon=child_taxon)
                
        if t.lower()==stoplevel.lower():
            # change stop flag
            stop = True

    return tree


def extract_fastx_from_bam(bamfile,genes,prefix,fastx='fastq'):
    '''
    Parameters:
    bamfile    read alignments
    genes      genes of interest
    prefix     prefix of output sequence file
    fastx      format of output sequence file
    '''
    # temporary bam file
    bamfile1 = os.path.join(tmp_aligns,genes[0]+'_'+str(len(genes))+'.bam')
    
    # extract to bam file
    cmd = ['samtools','view','-b',bamfile] + genes
    with open(bamfile1,'w') as f:
        call(cmd,stdout=f)

    # extract to fastq file
    fqfile = prefix+'.fastq'
    cmd = ['bedtools','bamtofastq','-i',bamfile1,'-fq',fqfile]
    call(cmd)
    
    fxfile = fqfile
    
    if fastx.lower() == 'fasta':
        fafile = prefix+'.fasta'
        cmd = ['seqtk','seq','-A',fqfile]
        with open(fafile,'w') as f:
            call(cmd,stdout=f)
        call(['rm','-f',fqfile])
        fxfile = fafile
        
    # cleanup 
    call(['rm','-f',bamfile1])
    
    return fxfile


def map_data_to_reference(sample_read,sample_metadata):
    scripts = []

    for idx,rec in sample_metadata.iterrows():
        # sample id
        sample = rec['SampleID']
        # filtered reads
        seqfile = sample_read.loc[sample_read.SampleID==sample,'Read'].squeeze()
        # alignment file
        samfile = os.path.join(tmp_aligns,sample+'_to_'+str(gg)+'.sam')
    
        # make up the command
        map_cmd = 'bowtie2 -x {0} -f -U {1} -S {2} --local -p {3} 2>/dev/null 1>/dev/null'\
                  .format(ggidx,seqfile,samfile,cpus)
        s2b_cmd = 'samtools view -F4 -bht {0}.fai {1} > {2}_temp.bam;\
samtools sort {2}_temp.bam {2} 2>/dev/null;\
samtools index {2}.bam 2>/dev/null;\
rm {2}_temp.bam'\
                  .format(ggref,samfile,samfile.replace('.sam',''))
    
        # write to the bash script
        script = os.path.join(tmp_scripts,sample+'_bowtie2.sh')
        with open(script,'w') as f:
            f.write('{0};{1}'.format(map_cmd, s2b_cmd))
        
        # save the script
        scripts += [script]

    # execute
    map(exec_script,scripts)

    # merge all alignments to one bam file
    onebam = os.path.join(tmp_aligns,'merged_alignments.bam')

    inbams = []
    for f in glob(os.path.join(tmp_aligns,'*_to_'+str(gg)+'.bam')):
        inbams += [f]

    if len(inbams)==1:
        cmd = ['ln','-fs'] + inbams + [onebam]
    else:
        cmd = ['samtools','merge',onebam] + inbams
    call(cmd)

    cmd = ['samtools','index',onebam]
    call(cmd)

    # get gene absolute abundance levels
    gene_abun_file = os.path.join(tmp_aligns,'gene_abun.txt')
    cmd = ['samtools','view','-F4',onebam,'|','cut','-f3','|','sort','--parallel',str(cpus),'|','uniq','-c',
           '>',gene_abun_file]
    exec_command(' '.join(cmd))
 
    # read gene absolute abundances
    gene_abun = pd.read_table(gene_abun_file,index_col=False,delim_whitespace=True,
                              names=['Abundance','Gene'],dtype={'Abundance':'float64','Gene':'S16'})

    return onebam,gene_abun


def search_candidate_clade(gene_tax_abun):
    # build taxonomic tree
    tree = build_taxonomic_tree(gene_tax_abun,'Genus')

    # data structure to store candidate clades
    clades = defaultdict(list)
    rare_clades = defaultdict(list)

    # get clade identity
    def clade_identity(clade):
        subclade_identity = defaultdict(list)

        # tree traversal
        for subclade in clade.traverse('preorder'):
            if len(subclade.taxon)>3:
                subclade_identity.setdefault(subclade.taxon[:3],[]).append(subclade.taxon)
            
        identity = clade.taxon
        # find the deepest common identity
        for level in phylolevels[::-1]:
            level = level[0].lower()+'__'
            if level in subclade_identity:
                if len(subclade_identity[level])==1:
                    identity = subclade_identity[level]
                    break
        
        return identity[0]

    # predefine new attribute
    for clade in tree.traverse('preorder'):
        clade.add_features(is_candidate=False)

    # traverse the tree
    for clade in tree.traverse('preorder'):
        if (not clade.is_root()) and \
           (not clade.taxon.startswith('k__')) and \
           (not clade.taxon.startswith('p__')):
            if not clade.up.is_candidate:
                is_candidate = False
                for child in clade.children:
                    if child.abundance<abs_abun_thresh:
                        continue
                    if len(child.taxon)==3:
                        is_candidate = True
                        break
                    elif child.abundance<total_abun*rel_abun_thresh:
                        is_candidate = True
                        break
            else:
                is_candidate = True
            if clade.is_leaf() and (clade.abundance>=abs_abun_thresh):
                is_candidate = True
            # save it if it is candidate
            if is_candidate and (not clade.up.is_candidate):
                clade_id = clade_identity(clade)
                if clade.abundance>total_abun*rel_abun_thresh:
                    clades.setdefault(clade_id,[]).append(clade)
                else:
                    rare_clades.setdefault(clade_id,[]).append(clade)
            # update clade state
            clade.is_candidate = is_candidate

    return clades,rare_clades



def read_vsearch_uc(ucfile):
    '''Just read uc records'''
    ucdata = pd.read_table(ucfile,index_col=False,header=None,
                           names=['RecordType','ClusterNumber','Size','Identity','Strand',
                                  'D1','D2','CompressedAlignment','Query','Target'])
    return ucdata

def parse_vsearch_uc(ucfile):
    '''Parse uc records and return clusters. clusters of zero members are deleted.'''
    ucdata = read_vsearch_uc(ucfile)
    
    def append_func(g):
        if g.shape[0]>1:
            return pd.DataFrame({'QuerySet':[g.Query.squeeze().tolist()],
                                 'QuerySetSize':[g.shape[0]]})
        else:
            return pd.DataFrame({'QuerySet':[[g.Query.squeeze()]],
                                 'QuerySetSize':[g.shape[0]]})
        
    uch = ucdata.loc[ucdata.RecordType=='H']\
                .groupby('Target')\
                .apply(append_func)\
                .reset_index(level=0)
    uch.index = xrange(uch.shape[0])
                
    return uch
    
def read_reassignment(readfile,ampliconfile,cores=1):
    '''re-assign reads to amplicons'''
    # dereplication
    readfile_derep_rs,readfile_derep,readfile_uc = get_uniq_reads(readfile,cores)
        
    # computing alignment
    samfile = amplicon_read_mapping(ampliconfile,readfile_derep,cores)
    
    # parse uc file
    derep_readset = parse_vsearch_uc(readfile_uc)
        
    # parse sam file
    amplicon_reads = defaultdict(list)
        
    samin = pysam.AlignmentFile(samfile,'r')
    for record in samin.fetch():
        if record.reference_id<0:
            continue
        amplicon = record.reference_name
        read = record.query_name
        if sum(derep_readset.Target==read)>0:
            amplicon_reads.setdefault(amplicon,[]).append(read)
            amplicon_reads.setdefault(amplicon,[]).extend(derep_readset\
                                                           .loc[derep_readset.Target==read,'QuerySet']\
                                                           .values[0])
    samin.close()
        
    call(['rm',samfile])
    return amplicon_reads
        
def get_subreads(prefix,readfile,readset):
    subreadfile = prefix+'.fa'
    cmd = ['samtools','faidx',readfile]+readset+['>',subreadfile]
    cmd += [';']
    cmd += ['samtools','faidx',subreadfile]
    exec_command(' '.join(cmd))
    return subreadfile

def get_gene_reads(prefix,geneset):
    readfile = extract_fastx_from_bam(onebam,geneset,prefix,'fasta')
    call(['samtools','faidx',readfile])
    return readfile

def get_repseq(readfile,cores=1):
    derep_readfile = os.path.splitext(readfile)[0]+'_derep.fa'
    derep_readuc = os.path.splitext(readfile)[0]+'_derep.uc'
    
    # get dereplicated reads
    derep_readset,derep_readfile,derep_readuc = get_uniq_reads(readfile,cores)
    
    # get representative sequence
    ucdata = read_vsearch_uc(derep_readuc)
    ucdata_centroids = ucdata.loc[ucdata.RecordType=='S']
    ucdata_centroids.sort_values('Size',inplace=True,ascending=False)
    repseqname = ucdata_centroids['Query'].values[0]
    repseqfile = os.path.join(os.path.dirname(readfile),repseqname+'_rep.fa')
    cmd = ['samtools','faidx',readfile,repseqname] + ['>',repseqfile]
    cmd += [';']
    cmd += ['samtools','faidx',repseqfile]
    exec_command(' '.join(cmd))
    return repseqname,repseqfile
    
def get_uniq_reads(readfile,cores=1):        
    uniqreadfile = os.path.splitext(readfile)[0]+'_uniq.fa'
    uniqreaduc = os.path.splitext(readfile)[0]+'_uniq.uc'
    cmd = '{0} --derep_fulllength {1} --output {2} --uc {3} --minuniquesize {4} --threads {5} 1>/dev/null 2>/dev/null'\
          .format(vsearch,readfile,uniqreadfile,uniqreaduc,derep_minsize,cores)
    exec_command(cmd)
    
    ucdata = read_vsearch_uc(uniqreaduc)
    ucdata_c = ucdata.loc[ucdata.RecordType=='C']
    if ucdata_c.shape[0]>1:
        uniqreadset = ucdata.loc[ucdata.RecordType=='C','Query'].squeeze().tolist()
    else:
        uniqreadset = list(ucdata.loc[ucdata.RecordType=='C','Query'].squeeze())
        
    return uniqreadset,uniqreadfile,uniqreaduc

def read_msa(readfile,cores):
    msafile = os.path.splitext(readfile)[0]+'_msa.fa'
    cmd = '{0} --cluster_fast {1} --id 0 --msaout {2} --threads {3} 1>/dev/null 2>/dev/null'\
          .format(vsearch,readfile,msafile,cores)
    exec_command(cmd)
    
    confile = os.path.splitext(readfile)[0]+'_cons.fa'
    # parse multiple sequence alignment
    msa = list(SeqIO.parse(msafile,'fasta'))
    # write out consensus sequence
    # make sure consensus sequence has no abnormal letters
    msa[-1].seq = Seq.Seq(str(msa[-1].seq).replace('+','-'))
    SeqIO.write(msa[-1],confile,'fasta')
    # write out read sequences
    msa[0].id = msa[0].id.replace('*','')
    msa[0].name = msa[0].name.replace('*','')
    msa[0].description = msa[0].description.replace('*','')
    SeqIO.write(msa[:-1],msafile,'fasta')
    return msafile,confile,msa[-1].name


def msa2bam(confile,msafile):
    '''confile: aligned consensus sequence
    msafile: aligned read sequences'''
    def ilen(it):
        return sum(1 for _ in it)

    def runlength_enc(xs):
        return ((x,ilen(gp)) for x, gp in its.groupby(xs))
    
    # compute cigar
    def cigar(target,query):
        tq = np.zeros(len(target),dtype=int)
        i=0
        for t,q in its.izip(target,query):
            if t=='-' and q=='-':
                tq = np.delete(tq,i)
                continue
            elif t=='-':
                tq[i] = 1
            elif q=='-':
                tq[i] = 2
            i += 1
        return list(runlength_enc(tq))
            
    # compute mismatch
    def mismatches(target,query):
        i = 0
        for t,q in its.izip(target,query):
            if t!='-' and q!='-':
                if t!=q:
                    i += 1
        return i
        
    # load alignments
    msa = AlignIO.read(msafile,'fasta')
    # consensus sequence
    consensus = SeqIO.read(confile,'fasta')
    
    # write bam file
    bamfile = msafile.replace('.fa','.bam')
    call(['rm','-rf',bamfile])
    
    # bam header
    header = {'HD':{'VN':'1.0'},
              'SQ':[{'LN':len(str(consensus.seq).replace('-','')),
                     'SN':consensus.name}]}
    
    # write alignment record
    with pysam.AlignmentFile(bamfile,'wb',header=header) as f:
        for i in xrange(len(msa)):
            a = pysam.AlignedSegment()
            a.query_name = msa[i].name.replace('*','')
            a.query_sequence = str(msa[i].seq).replace('-','')
            a.flag = 0
            a.reference_id = 0
            a.reference_start = 0
            a.mapping_quality = 40
            a.next_reference_id = -1
            a.next_reference_start = -1
            a.template_length = 0
            a.query_qualities = [40]*len(a.query_sequence)
            cig = cigar(consensus.seq,msa[i].seq)
            # modify cigar
            if cig[0][0]==1:
                cig[0]=(4,cig[0][1])
            elif cig[0][0]==2:
                # change starting position
                a.reference_start += cig[0][1]
                del cig[0]
            if cig[-1][0]==1:
                cig[-1]=(4,cig[-1][1])
            elif cig[-1][0]==2:
                del cig[-1]
            a.cigar = tuple(cig)
            a.tags = (('NM',mismatches(consensus.seq,msa[i].seq)),)
            f.write(a)
            
    call(['cp',bamfile,os.path.splitext(bamfile)[0]+'_temp.bam'])
    call(['samtools','sort',os.path.splitext(bamfile)[0]+'_temp.bam',os.path.splitext(bamfile)[0]])
    call(['samtools','index',bamfile])
    return bamfile
    


def amplicon_read_msa(ampliconfile,readfile,cores):
    # merge ampliconfile and readfile
    tempfile = os.path.splitext(ampliconfile)[0]+'_and_reads.fa'
    cmd = ['cat',ampliconfile,readfile,'>',tempfile]
    exec_command(' '.join(cmd))
    
    msafile = os.path.splitext(tempfile)[0]+'_msa.fa'
    # run msa
    cmd = '{0} --cluster_fast {1} --id 0 --msaout {2} --threads {3} 1>/dev/null 2>/dev/null'\
          .format(vsearch,tempfile,msafile,cores)
    exec_command(cmd)
    
    amplicon = SeqIO.read(ampliconfile,'fasta')
    confile = os.path.splitext(ampliconfile)[0]+'_aligned.fa'

    # parse msa sequence
    msa = list(SeqIO.parse(msafile,'fasta'))
    msa2 = []
    for m in msa:
        m.id = m.id.replace('*','')
        m.name = m.name.replace('*','')
        if m.name == amplicon.name:
            # make sure there is no abnormal sequences
            m.seq = Seq.Seq(str(m.seq).replace('+','-'))
            SeqIO.write(m,confile,'fasta')
        elif m.name == 'consensus':
            continue
        else:
            msa2 += [m]
            
    SeqIO.write(msa2,msafile,'fasta')
    
    return msafile,confile

def amplicon_read_mapping(ampliconfile,readfile,cores=1):                
    # make bowtie2 index
    cmd = 'bowtie2-build {0} {0} 1>/dev/null 2>/dev/null'.format(ampliconfile)
    exec_command(cmd)
        
    # map reads to amplicon
    samfile = os.path.splitext(ampliconfile)[0]+'.sam'
    cmd = 'bowtie2 -x {0} -f -U {1} -S {2} -p {3} --local 1>/dev/null 2>/dev/null'\
          .format(ampliconfile,readfile,samfile,cores)
    exec_command(cmd)
        
    return samfile

def is_empty_file(fn):
    if (not os.path.isfile(fn)) or (os.path.getsize(fn)==0):
        return True
    return False

def is_empty_bam(fn):
    return reduce(lambda x,y:x+y,[int(l.split('\t')[2]) for l in pysam.idxstats(fn).strip().split('\n')])==0
    

def amplicon_assembly(clade_amplicons,tmpdir,cores=1):
    # the number of amplicons at this moment
    N = clade_amplicons.shape[0]
    # list of old amplicons replaced by new amplicons
    to_delete = []
    # counting of new amplicons, zero based
    nc = 0
    for i in clade_amplicons.index:
        if clade_amplicons.loc[i,'Finish'] or clade_amplicons.loc[i,'Delete']:
            continue
            
        amplicon = clade_amplicons.loc[i,'Amplicon']
        readfile = clade_amplicons.loc[i,'ReadFile']
        readset = pd.read_table(readfile+'.fai',index_col=False,header=None,usecols=[0]).squeeze().tolist()
        #bam = clade_amplicons.loc[i,'ReadBam']
        sam = amplicon_read_mapping(amplicon,readfile,cpus)
        bam = sam.replace('.sam','.bam')
        cmd = 'samtools view -F4 -bht {0}.fai {1} > {2}_temp.bam;\
samtools sort {2}_temp.bam {2} 1>/dev/null 2>/dev/null;\
samtools index {2}.bam 1>/dev/null 2>/dev/null;'.format(amplicon,sam,bam.replace('.bam',''))
        exec_command(cmd)
        
        if is_empty_bam(bam):
            to_delete += [i]
            continue

        # run StrainCall
        amplicon_name,amplicon_len = pd.read_table(amplicon+'.fai',
                                                   index_col=False,
                                                   header=None,
                                                   usecols=[0,1]).squeeze().tolist()
        cmd = [straincall,amplicon,bam,
               '-r','\"{0}:1-{1}\"'.format(amplicon_name,amplicon_len),
               '-t','0.03','-D','60','-d',str(1-otu_thresh)]
        new_amplicon_seqfile = os.path.join(tmpdir,amplicon_name+'_new.fa')
        cmd += ['>',new_amplicon_seqfile]
        cmd += ['&&']
        cmd += ['samtools','faidx',new_amplicon_seqfile]
        exec_command(' '.join(cmd))
            
        # update global data structure
        new_amplicon_seqs = []
        new_amplicon_names = []
        new_amplicons = list(SeqIO.parse(new_amplicon_seqfile,'fasta'))
        nai = nc+N
        for new_amplicon in new_amplicons:
            outfile = os.path.join(tmpdir,'amplicon_'+str(nai)+'.fa')
            new_amplicon.id = 'amplicon_'+str(nai)
            new_amplicon.name = 'amplicon_'+str(nai)
            SeqIO.write(new_amplicon,outfile,'fasta')
            call(['samtools','faidx',outfile])
            new_amplicon_seqs += [outfile]
            new_amplicon_names += [new_amplicon.name]
            nai += 1
        SeqIO.write(new_amplicons,new_amplicon_seqfile,'fasta')
            
        new_amplicon_errorrates = []
        for new_amplicon in new_amplicons:
            errormodel = new_amplicon.description.split(';')
            errorrate = 0
            for e in errormodel:
                if e.startswith(('AA','CC','GG','TT')):
                    tag,prob = e.split(':')
                    errorrate += 1-np.float(prob)
            if errorrate > 0:
                new_amplicon_errorrates += [errorrate/4.]
            else:
                new_amplicon_errorrates += [error_rate]
                
        new_amplicon_readsets = []
        new_amplicon_readfiles = []
        new_amplicon_readbams = []
        new_amplicon_uniqreadfiles = []
        new_amplicon_uniqreaducs = []
        if len(new_amplicon_seqs) == 1:
            new_amplicon_readfiles = [readfile]
            new_amplicon_uniqreadfiles = [clade_amplicons.loc[i,"DerepReadFile"]]
            new_amplicon_uniqreaducs = [clade_amplicons.loc[i,"DerepReadUc"]]
            # compute new alignment
            msafile,confile = amplicon_read_msa(new_amplicon_seqs[0],
                                                new_amplicon_uniqreadfiles[0],
                                                cores)
            new_amplicon_readbams = [msa2bam(confile,msafile)]
            
        else:
            new_assignments = read_reassignment(readfile,new_amplicon_seqfile,cores)
            for new_amplicon in new_amplicons:
                new_amplicon_readsets += [new_assignments[new_amplicon.name]]
        
            for ampliconfile,subreadset in zip(new_amplicon_seqs,new_amplicon_readsets):
                prefix = os.path.splitext(ampliconfile)[0]+'_read'
                subreadfile = get_subreads(prefix,readfile,subreadset)
                uniqreadset,uniqreadfile,uniqreaduc = get_uniq_reads(subreadfile,cores)
                new_amplicon_readfiles += [subreadfile]
                new_amplicon_uniqreadfiles += [uniqreadfile]
                new_amplicon_uniqreaducs += [uniqreaduc]
                # compute new alignment
                if not is_empty_file(uniqreadfile):
                    msafile,confile = amplicon_read_msa(ampliconfile,uniqreadfile,cores)
                    new_amplicon_readbams += [msa2bam(confile,msafile)]
                else:
                    new_amplicon_readbams += [None]
            
        # 1. append new amplicons
        for k in xrange(len(new_amplicon_seqs)):
            j = nc+N+k
            clade_amplicons.loc[j] = [new_amplicon_seqs[k],
                                      new_amplicon_names[k],
                                      new_amplicon_errorrates[k],
                                      new_amplicon_readfiles[k],
                                      new_amplicon_readbams[k],
                                      new_amplicon_uniqreadfiles[k],
                                      new_amplicon_uniqreaducs[k],
                                      False,
                                      False]
        nc += len(new_amplicon_seqs)
        
        # mark the amplicons to be deleted
        to_delete += [i]
                
    # delete some amplicons
    for d in to_delete:
        clade_amplicons.loc[d,'Delete'] = True
    
    

def amplicon_partition(clade_amplicons,tmpdir,cores=1):
    global error_amplicon

    def poisson_test(L,e,x,N,n):
        from scipy.stats import poisson
        good = True
        # probability of generating the read
        lam = L*e
        p = 1-poisson.cdf(x,lam)+poisson.pmf(x,lam)
        # compute pvalue
        mu = p*N
        a = 1-poisson.pmf(0,mu)
        q = 1-poisson.cdf(n,mu)+poisson.pmf(n,mu)
        if a>0:
            q = q/a
        # pvalue adjustment
        q = q*N
        
        # check pvalue
        if q<poisson_thresh:
            good = False
        
        return good
        
    to_delete = []
    outlier_readfiles = []
    M = clade_amplicons.shape[0]
    # loop over amplicons
    for i in clade_amplicons.index:
        if clade_amplicons.loc[i,'Finish'] or clade_amplicons.loc[i,'Delete']:
            continue
            
        bamfile = clade_amplicons.loc[i,'DerepReadBam']
        if bamfile is None:
            to_delete += [i]
            continue

        ampliconfile = clade_amplicons.loc[i,'Amplicon']
        ampliconname = clade_amplicons.loc[i,'AmpliconName']
        errorrate = clade_amplicons.loc[i,'ErrorRate']
        ampliconreadfile = clade_amplicons.loc[i,'ReadFile']
        uniqreadfile = clade_amplicons.loc[i,'DerepReadFile']
        uniqreaduc = clade_amplicons.loc[i,'DerepReadUc']
        readset = pd.read_table(ampliconreadfile+'.fai',index_col=False,header=None,usecols=[0]).squeeze().tolist()        
        # parse alignment
        amplicon_outlier_reads = []
        amplicon_inlier_reads = []
        ucdata = parse_vsearch_uc(uniqreaduc)
        N = len(readset)
        have_outlier = False
        samin = pysam.AlignmentFile(bamfile,'rb')
        for record in samin.fetch(ampliconname):            
            qn = record.query_name
            L = record.reference_length
            e = errorrate
            x = record.get_tag('NM')
            if sum(ucdata.Target==qn)>0:
                n = ucdata.loc[ucdata.Target==qn,'QuerySetSize'].squeeze()+1
                                
                # poisson test
                good = poisson_test(L,e,x,N,n)
                if (not good) and (n>N*rel_abun_thresh) and (x>L*(1-otu_thresh)):
                    have_outlier = True
                    amplicon_outlier_reads += [qn] + ucdata.loc[ucdata.Target==qn,'QuerySet'].values[0]
                else:
                    amplicon_inlier_reads += [qn] + ucdata.loc[ucdata.Target==qn,'QuerySet'].values[0]
            if x>0:
                if error_amplicon.empty:
                    error_amplicon.loc[0] = [x,1]
                elif any(error_amplicon.ErrorAmpliconNum==x):
                    error_amplicon.loc[error_amplicon.ErrorAmpliconNum==x,'Frequency'] += 1
                else:
                    #error_amplicon.loc[error_amplicon.shape[0]] = [x,1]
                    error_amplicon = error_amplicon.append(pd.DataFrame([[x,1]],
                                                           columns=['ErrorAmpliconNum','Frequency']),
                                                           ignore_index=True)
        samin.close()
        
        # update amplicon record
        if have_outlier:
            prefix = os.path.splitext(ampliconfile)[0]+'_outlier'
            outlier_readfiles += [get_subreads(prefix,ampliconreadfile,amplicon_outlier_reads)]
            if len(amplicon_inlier_reads)>0:
                prefix = os.path.splitext(ampliconreadfile)[0]+'_inliner'
                inlier_readfile = get_subreads(prefix,ampliconreadfile,amplicon_inlier_reads)
                clade_amplicons.loc[i,'ReadFile'] = inlier_readfile
                inlier_uniqrs,inlier_uniqreadfile,inlier_uniqreaduc = get_uniq_reads(inlier_readfile,cores)
                clade_amplicons.loc[i,'DerepReadFile'] = inlier_uniqreadfile
                clade_amplicons.loc[i,'DerepReadUc'] = inlier_uniqreaduc
                # compute new alignment
                uniqmsafile,confile = amplicon_read_msa(clade_amplicons.loc[i,'Amplicon'],
                                                        clade_amplicons.loc[i,'DerepReadFile'],
                                                        cores)
                clade_amplicons.loc[i,'DerepReadBam'] = msa2bam(confile,uniqmsafile)
            else:
                to_delete += [i]
        else:
            clade_amplicons.loc[i,'Finish'] = True
        
    # create a new amplicon for outliers
    finished = True
    if len(outlier_readfiles)>0:
        finished = False
        outlier_readfile = os.path.join(tmpdir,'outlier_reads.fa')
        cmd = ['cat']+outlier_readfiles+['>',outlier_readfile,';']+['samtools','faidx',outlier_readfile]
        exec_command(' '.join(cmd))
        # unique reads
        outlier_uniqreadset,outlier_uniqreadfile,outlier_uniqreaduc = get_uniq_reads(outlier_readfile,cores)
        # 
        outlier_uniqreadmsa,outlier_repseqfile,outlier_repseqname = read_msa(outlier_uniqreadfile,cores)
        # normalize sequence name
        outlier_repseqname = 'amplicon_'+str(M)
        outlier_repseq = SeqIO.read(outlier_repseqfile,'fasta')
        outlier_repseq.id = outlier_repseqname
        outlier_repseq.name = outlier_repseqname
        SeqIO.write(outlier_repseq,outlier_repseqfile,'fasta')
        # remove gap letter
        outlier_repseq.seq = Seq.Seq(str(outlier_repseq.seq).replace('-',''))
        outlier_repseqfile = os.path.join(tmpdir,'amplicon_'+str(M)+'.fa')
        SeqIO.write(outlier_repseq,outlier_repseqfile,'fasta')
        call(['samtools','faidx',outlier_repseqfile])
        # compute new alignment
        outlier_uniqreadbam = msa2bam(outlier_repseqfile,outlier_uniqreadmsa)
        
        clade_amplicons.loc[M] = [outlier_repseqfile,
                                  outlier_repseqname,
                                  error_rate,
                                  outlier_readfile,
                                  outlier_uniqreadbam,
                                  outlier_uniqreadfile,
                                  outlier_uniqreaduc,
                                  False,
                                  False]
    
    for d in to_delete:
        clade_amplicons.loc[d,'Delete'] = True
    
    return finished
    
def clade_amplicon_assembly(identity,clade,cores=1):
    if verbose:
        logging.info('assembling {}'.format(identity))

    # create a temporary directory for clade
    tmpdir = os.path.join(tmp_clusters,identity.replace(' ','_'))
    call(['mkdir','-p',tmpdir])
    os.chdir(tmpdir)
    
    # get gene set
    geneset = []
    for c in clade:
        lineage = c.name.split(';')
        geneset += gene_tax_abun.groupby(phylolevels[:len(lineage)])\
                                .get_group(tuple(lineage))['Gene'].tolist()
    
    # global data structure
    clade_amplicons = pd.DataFrame(columns=['Amplicon','AmpliconName','ErrorRate',
                                            'ReadFile',
                                            'DerepReadBam',
                                            'DerepReadFile','DerepReadUc',
                                            'Finish','Delete'])
        
    # prepare representative sequence and reads
    prefix = os.path.join(tmpdir,'readfile')
    readfile = get_gene_reads(prefix,geneset)
    uniqreadset,uniqreadfile,uniqreaduc = get_uniq_reads(readfile,cores)

    # check whether uniqreadfile is empty
    #if (not os.path.isfile(uniqreadfile)) or (os.path.getsize(uniqreadfile)==0):
    if is_empty_file(uniqreadfile):
        return None

    # multiple sequence alignment
    uniqreadmsafile,repseqfile,repseqname = read_msa(uniqreadfile,cores)
        
    # normalize sequence name
    repseqname = 'amplicon_0'
    repseq = SeqIO.read(repseqfile,'fasta')
    repseq.id = repseqname
    repseq.name = repseqname
    SeqIO.write(repseq,repseqfile,'fasta')
    
    # bam and unique reads
    uniqreadbamfile = msa2bam(repseqfile,uniqreadmsafile)
    
    # remove gaps
    repseq.seq = Seq.Seq(str(repseq.seq).replace('-',''))
    repseqfile = os.path.join(tmpdir,repseqname+'.fa')
    SeqIO.write(repseq,repseqfile,'fasta')
    call(['samtools','faidx',repseqfile])
            
    # initialize the global data structure
    clade_amplicons.loc[0] = [repseqfile,repseqname,error_rate,
                              readfile,
                              uniqreadbamfile,
                              uniqreadfile,uniqreaduc,
                              False,
                              False]
    
    # loop
    finished = False
    while (not finished):
        # assembly
        amplicon_assembly(clade_amplicons,tmpdir,cores)
        
        # partitioning
        finished = amplicon_partition(clade_amplicons,tmpdir,cores) 
     
    # back to cwd
    os.chdir(cwd)
    
    amplicon_seqs = []
    id = 0
    for idx,rec in clade_amplicons.iterrows():
        if rec['Delete']:
            continue
        recseq = SeqIO.read(rec['Amplicon'],'fasta')
        recseq.id = identity+"_"+str(id)
        recseq.name = identity+"_"+str(id)
        amplicon_seqs += [recseq]
        id += 1
        
    amplicon_output = os.path.join(tmpdir,identity+"_amplicons.fa")
    SeqIO.write(amplicon_seqs,amplicon_output,'fasta')
    
    return amplicon_output


prehelp='''
The program AMASS is to assemble 16S amplicon sequencing reads into 
high-quality OTU consensus sequences.

'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=prehelp,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('sample_read',help='a list of sample reads, one sample per line')
    parser.add_argument('sample_metadata',help='a list of sample metadatas, one sample per line')
    parser.add_argument('parameter_config',help='parameter settings')
    parser.add_argument('-v',dest='verbose',action='store_true',default=False,help='verbose output')

    opts = parser.parse_args()
    opts.sample_read = os.path.abspath(opts.sample_read)
    opts.sample_metadata = os.path.abspath(opts.sample_metadata)
    opts.parameter_config = os.path.abspath(opts.parameter_config)
    verbose = opts.verbose

    # parse parameters
    parse_parameters(opts.parameter_config)

    # set log reporter
    logging.basicConfig(format='[%(asctime)s] %(levelname)s : %(message)s', level=logging.INFO)

    # parse sample read file
    sample_read = parse_sample_read(opts.sample_read)

    # parse sample metadata file
    sample_metadata = parse_sample_metadata(opts.sample_metadata)

    # load gene index
    gene_bed = load_gene_index(ggref)

    # load gene taxonomy
    if gg.lower()=='greengenes':
        gene_tax = load_greengene_tax(ggtax)

    # compute abundance
    if verbose:
        logging.info('map reads to reference sequences and compute abundance levels')
    onebam,gene_abun = map_data_to_reference(sample_read,sample_metadata)
    # total abundance
    total_abun = gene_abun['Abundance'].sum()

    # merge gene abundance and gene taxonomy
    gene_tax_abun = pd.merge(gene_abun,gene_tax,on='Gene')

    # search candidate clades
    if verbose:
        logging.info('search taxonomic tree to find candidate clades')
    clades,rare_clades = search_candidate_clade(gene_tax_abun)

    # assemble amplicon sequences
    if verbose:
        logging.info('there are {} abundant candidate clades'.format(len(clades)))

    ampliconfiles = []
    for identity,subclades in clades.iteritems():
        amp = clade_amplicon_assembly(identity,subclades,cpus)
        if amp is not None:
            ampliconfiles += [amp]

    if len(error_amplicon['Frequency']):
        error_amplicon['Prob'] = error_amplicon['Frequency']/error_amplicon['Frequency'].sum()
    else:
        error_amplicon['Prob'] = error_amplicon['Frequency']

    # process rare clades
    if verbose:
        logging.info('there are {} rare candidate clades'.format(len(rare_clades)))

    for identity,subclades in rare_clades.iteritems():
        abun = sum([c.abundance for c in subclades])
        pvalue = error_amplicon.loc[error_amplicon.ErrorAmpliconNum>=abun,'Prob'].sum()
        if pvalue<rare_test_p:
            amp = clade_amplicon_assembly(identity,subclades,cpus)
            if amp is not None:
                ampliconfiles += [amp]
        elif verbose:
            logging.info('{} does not pass rare testing'.format(identity))

    # merge together
    oneampliconfile0 = os.path.join(cwd,'all_amplicons0.fa')
    cmd = 'cat {0} > {1}'.format(' '.join(ampliconfiles),oneampliconfile0)
    exec_command(cmd)

    oneampliconfile = os.path.join(cwd,'all_amplicons.fa')
    cmd = '{0} --derep_fulllength {1} --output {2} 1>/dev/null 2>/dev/null && rm {1}'\
          .format(vsearch,oneampliconfile0,oneampliconfile)
    exec_command(cmd)

