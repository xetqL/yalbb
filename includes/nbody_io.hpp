//
// Created by xetql on 2/5/18.
//

#ifndef NBMPI_NBODY_IO_HPP
#define NBMPI_NBODY_IO_HPP

#include <vector>
#include <cstdio>
#include <cstdint>
#include <arpa/inet.h>
#include <vector>

//TODO: REWRITE THE DATA FRAME WRITER MODULE IN A EASIEST AND C++er WAY. Then, in Python create a viewer.


/*@T
 * \section{Binary output}
 *
 * There are two output file options for our code: text and binary.
 * Originally, I had only text output; but I got impatient waiting
 * to view some of my longer runs, and I wanted something a bit more
 * compact, so added a binary option
 *
 * The viewer distinguishes the file type by looking at the first few
 * characters in the file: the tag [[NBView00]] means that what
 * follows is text data, while the tag [[NBView01]] means that what
 * follows is binary.  If you care about being able to read the
 * results of your data files across multiple versions of a code, it's
 * a good idea to have such a tag!
 *@c*/
#define VERSION_TAG "NBView01"

/*@T
 * Different platforms use different byte orders.  The Intel Pentium
 * hardware is little-endian, which means that it puts the least-significant
 * byte first -- almost like if we were to write a hundred twenty as 021 rather
 * than 120.  The Java system (which is where our viewer lives) is big-endian.
 * Big-endian ordering is also the so-called ``wire standard'' for sending
 * data over a network, so UNIX provides functions [[htonl]] and [[htons]]
 * to convert long (32-bit) and short (16-bit) numbers from the host
 * representation to the wire representation.  There is no corresponding
 * function [[htonf]] for floating point data, but we can construct
 * such a function by pretending floats look like 32-bit integers ---
 * the byte shuffling is the same.
 *@c*/
uint32_t htonf(void* data)
{
    return htonl(*(uint32_t*) data);
}

/*@T
 *
 * The header data consists of a count of the number of balls (a 32-bit
 * integer) and a scale parameter (a 32-bit floating poconst int number).
 * The scale parameter tells the viewer how big the view box is supposed
 * to be in the coordinate system of the simulation; right now, it is
 * always set to be 1 (i.e. the view box is $[0,1] \times [0,1]$)
 *@c*/
void write_header(FILE* fp, const int n)
{
    float scale = 1.0;
    uint32_t nn = htonl((uint32_t) n);
    uint32_t nscale = htonf(&scale);
    fprintf(fp, "%s\n", VERSION_TAG);
    fwrite(&nn,     sizeof(nn),     1, fp);
    fwrite(&nscale, sizeof(nscale), 1, fp);
}

/*@T
 *
 * The header data consists of a count of the number of balls (a 32-bit
 * integer) and a scale parameter (a 32-bit floating poconst int number).
 * The scale parameter tells the viewer how big the view box is supposed
 * to be in the coordinate system of the simulation; right now, it is
 * always set to be 1 (i.e. the view box is $[0,1] \times [0,1]$)
 *@c*/
void write_header(FILE* fp, const int n, float simsize)
{
    uint32_t nn = htonl((uint32_t) n);
    uint32_t nscale = htonf(&simsize);
    fprintf(fp, "%s\n", VERSION_TAG);
    fwrite(&nn,     sizeof(nn),     1, fp);
    fwrite(&nscale, sizeof(nscale), 1, fp);
}

void write_header(FILE* fp, const int n, double simsize)
{
    uint32_t nn = htonl((uint32_t) n);
    uint32_t nscale = htonf(&simsize);
    fprintf(fp, "%s\n", VERSION_TAG);
    fwrite(&nn,     sizeof(nn),     1, fp);
    fwrite(&nscale, sizeof(nscale), 1, fp);
}
/*@T
 *
 * After the header is a sequence of frames, each of which contains
 * $n_{\mathrm{particles}}$ pairs of 32-bit int floating point numbers.
 * There are no markers, end tags, etc; just the raw data.
 * The [[write_frame_data]] routine writes $n$ pairs of floats;
 * note that writing a single frame of output may involve multiple
 * calls to [[write_frame_data]]
 * Frame data just consists
 * integer) and a scale parameter (a 32-bit floating point number).
 * The scale parameter tells the viewer how big the view box is supposed
 * to be in the coordinate system of the simulation; right now, it is
 * always set to be 1 (i.e. the view box is $[0,1] \times [0,1]$)
 *@c*/
void write_frame_data(FILE* fp, const int n, float* x)
{
    for (int i = 0; i < n; ++i) {
        uint32_t xi = htonf(x++);
        uint32_t yi = htonf(x++);

        fwrite(&xi, sizeof(xi), 1, fp);
        fwrite(&yi, sizeof(yi), 1, fp);
    }
}

/**
 * Please accept this silly solution as it is and do not ask any question.
 * I gave up..
 * @param fp
 * @param n
 * @param els
 */
template<int N>
void write_frame_data(FILE* fp, int n, elements::Element<N> *els)
{
    float* dont_know_why = new float[N * n]; //x & y for each particle
    for(int i = 0; i < n; ++i) {
        dont_know_why[N*i]   = els[i].position[0];
        dont_know_why[N*i+1] = els[i].position[1];
        if(N==3) dont_know_why[N*i+2] = els[i].position[2];
    }
    write_frame_data(fp, n, dont_know_why);
    delete[] dont_know_why;
}

void write_header_bin(std::ofstream &stream, const int n, const int dimension, double nscale)
{
    stream.write(reinterpret_cast<const char*>(&n),        sizeof(int));
    stream.write(reinterpret_cast<const char*>(&dimension),sizeof(int));
    stream.write(reinterpret_cast<const char*>(&nscale),   sizeof(double));
}

template<int N>
void write_frame_data_bin(std::ofstream &stream, std::vector<elements::Element<N>>& els)
{
    for(elements::Element<N> &el : els ) {
        stream.write(reinterpret_cast<const char *>(&el.position[0]), sizeof(double));
        stream.write(reinterpret_cast<const char *>(&el.position[1]), sizeof(double));
        if (N == 3) stream.write(reinterpret_cast<const char *>(&el.position[0]), sizeof(double));
    }
}

struct SimpleXYZFormatter {
    template<int N>
    inline void write_data(std::ofstream &stream, elements::Element<N>& el){
        if(N<3) stream << el.position[0] << " " << el.position[1] << std::endl;
        else stream << el.position[0] << " " << el.position[1] << " " <<  el.position[2] << std::endl;
    }
    inline void write_header(std::ofstream &stream, const int n, float simsize){
        configure_stream(stream);
    }
    template<int N>
    inline void write_frame_header(std::ofstream &stream, std::vector<elements::Element<N>>& els, const sim_param_t* params){
        stream << els.size() << std::endl << "Lattice=\""<<0.0<<" "<<0.0<<" "<<0.0<<" "<<params->simsize<<" "<<params->simsize<<" "<<params->simsize<<"\""<< std::endl;
    }
private:
    inline void configure_stream(std::ofstream &stream, int precision = 6){
        stream << std::fixed << std::setprecision(6);
    }
};

struct SimpleCSVFormatter {
    const char separator;
    SimpleCSVFormatter(char separator) : separator(separator){}

    template<int N>
    inline void write_data(std::ofstream &stream, elements::Element<N>& el){
        stream << el.position[0] << separator << el.position[1];
        if(N > 2) stream << separator <<  el.position[2];
        stream << std::endl;
    }
    inline void write_header(std::ofstream &stream, const int n, float simsize){
        configure_stream(stream);
    }
    template<int N>
    inline void write_frame_header(std::ofstream &stream, std::vector<elements::Element<N>>& els, const sim_param_t* params){
        stream << "x coord" << separator << "y coord";
        if(N > 2) stream << separator << "z coord";
        stream << std::endl;
    }
private:
    inline void configure_stream(std::ofstream &stream, int precision = 6){
        stream << std::fixed << std::setprecision(6);
    }
};

template<int N, class FrameFormatter>
void write_frame_data(std::ofstream &stream, std::vector<elements::Element<N>>& els, FrameFormatter& formatter, const sim_param_t* params) {
    formatter.write_frame_header(stream, els, params);
    for(elements::Element<N> &el : els ) {
        formatter.write_data(stream, el);
    }
}

#endif //NBMPI_NBODY_IO_HPP
