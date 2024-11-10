# Lab 3: Hardware GEMM in Darknet

In this assignment you will develop a approximately timed GEMM fixed point accelerator for the Zynq platform and integrate it with darknet. In this version, the accelerator will only perform GEMM and will only be a target on the bus (just like the vector processor). Project will explore more advanced options. 

Common: see overview [Xilinx ISPD 2018](http://www.ispd.cc/slides/2018/s2_3.pdf)

## 1. Overview 

The instructions for ths lab are detailed in the following steps:

 1. (Reserved for feedback branch pull request. You will receive top level feedback there). 
 2. [Getting Started](.github/STARTING_ISSUES/2.%20Getting%20Started.md)
 3. [GEMM ACC](.github/STARTING_ISSUES/3.%20GEMM%20ACC.md)
 4. [Tiling](.github/STARTING_ISSUES/4.%20Tiling.md)
 5. [Darknet Integration](.github/STARTING_ISSUES/5.%20Darknet%20Integration.md)
 6. [Optimization](.github/STARTING_ISSUES/6.%20Optimization.md)


## 2. Register Interface 

At the end of this lab the full gemm processor's register map should be:

| **Address Space Offset** | **Name**  | **Description**                                                                                                                                                      |
|----------------------|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0x00 | CSR | The control and status register  |
| 0x04 | PC  | Current Descriptor / Instruction (Read Only) starting with 1 |
| 0x08 | 01_MA_START_ADDR | Start address of MA|
| 0x0C | 01_MB_START_ADDR | Start address of MB| 
| 0x10 | 01_MC_START_ADDR | Start address of MC|
| 0x14 | 01_MA_NR_ROW     | Number of rows in MA | 
| 0x18 | 01_MB_NR_ROW     | Number of rows in MB and number of columns in MA | 
| 0x1C | 01_MB_NR_COL     | Number of columns in MB (Note that MC's dimensions will be [01_MA_NR_ROW][01_MB_NR_COL])| 
| 0x20 | 01_INST_NEXT     | reference to next instruction (set to 0)|
| 0x24 ... | 02_*   | second descriptor |
| 0x100 - 0x800FF| DATA  | GemmProc ScratchPad (512KByte) |


The set of MMRs starting with `01_` capture one descriptor instructing the GEMM processor where to find the input and output data and their dimensions. 
Input / output data is stored in the scratchpad of the accelerator. 
The addresses (e.g. `01_MA_START_ADDR`) are in the DATA scratchpad address space in bytes. For example, assume MA was written starting to MMR offset 0x100 (i.e. the first address in the scratchpad) it would end up in the scratch pad offset 0. Consequently, `01_MA_START_ADDR` should be 0. `01_INST_NEXT` is reserved for a future extension when the processor supports multiple consequtive descriptors (instructions). For now, always set it to 0 indicating there is no next instruction and the processor will go to idle. 

![Matrix Multiply Dimensions](.github/matDims.png)

General principle of SW interaction:

1. Driver writes matrices into ScratchPad memory (DATA)
2. Driver writes descriptor (i.e. all the 01_* MMRs)
3. Driver initiates processing by writing to CSR
4. Driver polls CSR to wait until ACC is done processing

After accepting this assignment in github classroom, each step is converted into a [github issue](https://docs.github.com/en/issues). Follow the issues in numerically increasing issue number (the first issue is typically on the bottom of the list). 

## General Rules

Please commit your code frequently or at e very logical break. Each commit should have a meaningful commit message and [cross reference](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/autolinked-references-and-urls#issues-and-pull-requests) the issue the commit belongs to. Ideally, there would be no commits without referencing to a github issue. 

Please comment on each issue with the problems faced and your approach to solve them. Close an issue when done. 



