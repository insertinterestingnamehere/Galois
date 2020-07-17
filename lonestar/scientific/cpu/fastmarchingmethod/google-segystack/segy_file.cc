// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#if defined(__GNUC__)
_Pragma("GCC diagnostic ignored \"-Wunused-const-variable\"")
#endif

#include "segy_file.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <sstream>
#include <utility>
#include <cassert>

// #include "logging.h"
#include "galois/gIO.h"

namespace segystack {

std::ostream& operator<<(std::ostream& os,
                         const segystack::SegyFile::TextHeader& hdr) {
  os << hdr.toString();
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const segystack::SegyFile::BinaryHeader& hdr) {
  hdr.print(os);
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const segystack::SegyFile::Trace::Header& hdr) {
  hdr.print(os);
  return os;
}

namespace {

constexpr int kSegyTextHeaderLineWidth = 80;

struct BFHNumTracesPerEnsemble {
  static constexpr int offset = 13;
  typedef uint16_t type;
  static constexpr char desc[] = "Number of data traces per ensemble";
};
struct BFHNumAuxTracesPerEnsemble {
  static constexpr int offset = 15;
  typedef uint16_t type;
  static constexpr char desc[] = "Number of auxiliary data traces per ensemble";
};
struct BFHSampleInterval {
  static constexpr int offset = 17;
  typedef uint16_t type;
  static constexpr char desc[] = "Sample interval in microseconds(us)";
};
struct BFHNumSamplesPerTrace {
  static constexpr int offset = 21;
  typedef uint16_t type;
  static constexpr char desc[] = "Number of samples per data trace";
};
struct BFHSampleFormatCode {
  static constexpr int offset = 25;
  typedef uint16_t type;
  static constexpr char desc[] = "Data sample format code";
  // encoding
  static constexpr int IBM  = 1;
  static constexpr int IEEE = 5;
  // ...
};
struct BFHEnsembleFold {
  static constexpr int offset = 27;
  typedef uint16_t type;
  static constexpr char desc[] = "Ensemble fold - The expected number of data "
                                 "traces per ensemble (e.g. the CMP fold)";
};
struct BFHTraceSortingCode {
  static constexpr int offset = 29;
  typedef int16_t type;
  static constexpr char desc[] = "Trace sorting code (i.e. type of ensemble)";
  // encoding
  static constexpr int Other      = -1;
  static constexpr int Unknown    = 0;
  static constexpr int AsRecorded = 1;
  // ...
};
struct BFHMsmSys {
  static constexpr int offset = 55;
  typedef uint16_t type;
  static constexpr char desc[] = "Measurement system";
  // encoding
  static constexpr int Meters = 1;
  static constexpr int Feet   = 2;
  // ...
};
struct BFHRevMajor {
  static constexpr int offset = 301;
  typedef uint8_t type;
};
struct BFHRevMinor {
  static constexpr int offset = 302;
  typedef uint8_t type;
};
struct BFHTraceFlag {
  static constexpr int offset = 303;
  typedef uint16_t type;
  static constexpr char desc[] = "Fixed length trace flag";
  // encoding
  static constexpr int Same    = 1;
  static constexpr int MayVary = 0;
};
struct BFHNumExtTFH {
  static constexpr int offset = 305;
  typedef int16_t type;
  static constexpr char desc[] =
      "Number of 3200-byte, Extended Textual File Header records (-1 indicates "
      "variable number)";
};

///

struct THSeqNumInLine {
  static constexpr int offset = 1;
  typedef uint32_t type;
  static constexpr char desc[] = "Trace sequence number within line";
};
struct THOrigFieldRecNum {
  static constexpr int offset = 9;
  typedef uint32_t type;
  static constexpr char desc[] = "Original field record number";
};
struct THNumInRec {
  static constexpr int offset = 13;
  typedef uint32_t type;
  static constexpr char desc[] =
      "Trace number within the original field record";
};
struct THIDCode {
  static constexpr int offset = 29;
  typedef uint16_t type;
  static constexpr char desc[] = "Trace identification code";
  // encoding
  static constexpr int SeismicData = 1;
  // ...
};
struct THRecvGrpElev {
  static constexpr int offset = 41;
  typedef int32_t type;
  static constexpr char desc[] =
      "Receiver group elevation (w.r.t. Vertical datum)";
};
struct THSurfElevAtSrc {
  static constexpr int offset = 45;
  typedef int32_t type;
  static constexpr char desc[] = "Surface elevation at source";
};
struct THSrcDptBelowSurf {
  static constexpr int offset = 49;
  typedef int32_t type;
  static constexpr char desc[] = "Source depth below surface";
};
struct THDatumElevAtRecvGrp {
  static constexpr int offset = 53;
  typedef int32_t type;
  static constexpr char desc[] = "Datum elevation at receiver group";
};
struct THDatumElevAtSrc {
  static constexpr int offset = 57;
  typedef int32_t type;
  static constexpr char desc[] = "Datum elevation at source";
};
struct THWaterDptAtSrc {
  static constexpr int offset = 61;
  typedef int32_t type;
  static constexpr char desc[] = "Water depth at source";
};
struct THWaterDptAtRecvGrp {
  static constexpr int offset = 65;
  typedef int32_t type;
  static constexpr char desc[] = "Water depth at receiver group";
};
struct THElevDptScalar {
  static constexpr int offset = 69;
  typedef int16_t type;
  static constexpr char desc[] = "Scalar to be applied to all elevations and "
                                 "depths to give the real value";
};
struct THCoordScalar {
  static constexpr int offset = 71;
  typedef int16_t type;
  static constexpr char desc[] =
      "Scalar to be applied to all coordinates to give the real value";
};
struct THNumSamples {
  static constexpr int offset = 115;
  typedef uint16_t type;
  static constexpr char desc[] = "Number of samples in this trace";
};
struct THSampleInterval {
  static constexpr int offset = 117;
  typedef uint16_t type;
  static constexpr char desc[] =
      "Sample interval in microseconds (us) for this trace";
};

const std::vector<int> kSegyHeaderXCoordCandidateOffsets = {
    181, // X coordinate of ensemble (CDP) position of this trace.
    73,  // Source coordinate - X.
    201, // Non-standard but often used.
};
const std::vector<int> kSegyHeaderYCoordCandidateOffsets = {
    185, // Y coordinate of ensemble (CDP) position of this trace.
    77,  // Source coordinate - Y.
    205, // Non-standard but often used.
};
struct THEnsmX {
  static constexpr int offset = 181;
  typedef uint32_t type;
  static constexpr char desc[] =
      "X coordinate of ensemble (CDP) position of this trace (scaled)";
};
struct THEnsmY {
  static constexpr int offset = 185;
  typedef uint32_t type;
  static constexpr char desc[] =
      "Y coordinate of ensemble (CDP) position of this trace (scaled)";
};
struct THSrcX {
  static constexpr int offset = 73;
  typedef uint32_t type;
  static constexpr char desc[] = "Source coordinate - X";
};
struct THSrcY {
  static constexpr int offset = 77;
  typedef uint32_t type;
  static constexpr char desc[] = "Source coordinate - Y";
};
struct THShotpointNumScalar {
  static constexpr int offset = 201;
  typedef int16_t type;
  static constexpr char desc[] =
      "Scalar to be applied to the shotpoint number to give the real value";
};
struct THTransConstMantissa {
  static constexpr int offset = 205;
  typedef int32_t type;
  static constexpr char desc[] =
      "The mantissa as a two's complement integer of Transduction Constant - "
      "The multiplicative constant used to convert the Data Trace samples to "
      "Transduction Units";
};
struct THTransConstExponent {
  static constexpr int offset = 209;
  typedef int16_t type;
  static constexpr char desc[] =
      "The power of ten exponent as a two's complement integer of Transduction "
      "Constant - The multiplicative constant used to convert the Data Trace "
      "samples to Transduction Units";
};
struct THTransUnitsCode {
  static constexpr int offset = 211;
  typedef int16_t type;
  static constexpr char desc[] =
      "Transduction units - The unit of measurement of the Data Trace samples "
      "after they have been multiplied by the Transduction Constant";
  // encoding
  static constexpr int Unknown = 0;
  // ...
};

const std::vector<int> kSegyHeaderInlineCandidateOffsets = {
    189, // For 3-D poststack data, this field should be used for the in-line
         // number.
    213, // Non-standard but often used.
    9,   // Original field record number.
};
const std::vector<int> kSegyHeaderCrosslineCandidateOffsets = {
    193, // For 3-D poststack data, this field should be used for the
         // cross-line number.
    217, // Non-standard but often used.
    21,  // Ensemble number (i.e. CDP, CMP, CRP, etc).
};
struct TH3DPoststackInlineNum {
  static constexpr int offset = 189;
  typedef uint32_t type;
  static constexpr char desc[] = "For 3-D poststack data, this field should be "
                                 "used for the in-line number";
};
struct TH3DPoststackCrlineNum {
  static constexpr int offset = 193;
  typedef uint32_t type;
  static constexpr char desc[] = "For 3-D poststack data, this field should be "
                                 "used for the cross-line number";
};
struct THDeviceID {
  static constexpr int offset = 213;
  typedef uint16_t type;
  static constexpr char desc[] = "The unit number or id number of the device "
                                 "associated with the Data Trace";
};
struct THSrcTyAndOriCode {
  static constexpr int offset = 217;
  typedef int16_t type;
  static constexpr char desc[] = "Source Type/Orientation - Defines the type "
                                 "and the orientation of the energy source";
  // encoding
  static constexpr int Unknown = 0;
  // ...
};
struct THEnsmNum {
  static constexpr int offset = 21;
  typedef uint32_t type;
  static constexpr char desc[] = "Ensemble number (i.e. CDP, CMP, CRP, etc)";
};

// https://www.ibm.com/support/knowledgecenter/SS2MB5_14.1.0/com.ibm.xlf141.bg.doc/language_ref/asciit.html
const std::map<unsigned char, char> kEBCDICtoASCIImap = {
    {75, '.'},  {76, '<'},   {77, '('},  {78, '+'},  {79, '|'},  {80, '&'},
    {90, '!'},  {91, '$'},   {92, '*'},  {93, ')'},  {94, ';'},  {96, '-'},
    {97, '/'},  {106, '|'},  {107, ','}, {108, '%'}, {109, '_'}, {110, '>'},
    {111, '?'}, {121, '`'},  {122, ':'}, {123, '#'}, {124, '@'}, {125, '\''},
    {126, '='}, {127, '"'},  {129, 'a'}, {130, 'b'}, {131, 'c'}, {132, 'd'},
    {133, 'e'}, {134, 'f'},  {135, 'g'}, {136, 'h'}, {137, 'i'}, {145, 'j'},
    {146, 'k'}, {147, 'l'},  {148, 'm'}, {149, 'n'}, {150, 'o'}, {151, 'p'},
    {152, 'q'}, {153, 'r'},  {161, '~'}, {162, 's'}, {163, 't'}, {164, 'u'},
    {165, 'v'}, {166, 'w'},  {167, 'x'}, {168, 'y'}, {169, 'z'}, {192, '{'},
    {193, 'A'}, {194, 'B'},  {195, 'C'}, {196, 'D'}, {197, 'E'}, {198, 'F'},
    {199, 'G'}, {200, 'H'},  {201, 'I'}, {208, '}'}, {209, 'J'}, {210, 'K'},
    {211, 'L'}, {212, 'M'},  {213, 'N'}, {214, 'O'}, {215, 'P'}, {216, 'Q'},
    {217, 'R'}, {224, '\\'}, {226, 'S'}, {227, 'T'}, {228, 'U'}, {229, 'V'},
    {230, 'W'}, {231, 'X'},  {232, 'Y'}, {233, 'Z'}, {240, '0'}, {241, '1'},
    {242, '2'}, {243, '3'},  {244, '4'}, {245, '5'}, {246, '6'}, {247, '7'},
    {248, '8'}, {249, '9'}};

char getASCIIForEBCDIC(char c) {
  if (kEBCDICtoASCIImap.find(c) != kEBCDICtoASCIImap.end())
    return kEBCDICtoASCIImap.at(c);
  return ' ';
}

bool isTextInEBCDICFormat(const char* text, size_t length) {
  int alnumASCII = 0;
  for (size_t i = 0; i < length; i++) {
    if (std::isalnum(text[i]))
      alnumASCII++;
  }

  int alnumEBCDIC = 0;
  for (size_t i = 0; i < length; i++) {
    if (std::isalnum(getASCIIForEBCDIC(text[i])))
      alnumEBCDIC++;
  }

  if (alnumASCII > alnumEBCDIC)
    return false;
  return true;
}
} // namespace

const size_t SegyFile::TEXT_HEADER_SIZE;
const size_t SegyFile::BINARY_HEADER_SIZE;
const size_t SegyFile::TRACE_HEADER_SIZE;

SegyFile::SegyFile(const std::string& filename)
    : text_header_(nullptr), binary_header_(nullptr), num_samples_per_trc_(0),
      first_hdr_ptr_(nullptr), hdr_ptr_(nullptr), trc_ptr_(nullptr),
      cur_offset_(0) {
  file_ = MmapFile::Create(filename);
  open();
}
SegyFile::~SegyFile() { close(); }

std::map<SegyFile::Trace::Header::Attribute, int>
SegyFile::guessTraceHeaderOffsets() const {
  if (!is_open()) {
    throw std::runtime_error("File " + name() + " not opened for reading!");
  }

  std::uint64_t prev_offset = cur_offset_;

  Trace trace1, trace2;
  seek(0);
  read(trace1);
  seek(1);
  read(trace2);

  // restore back to where we were.
  seek(prev_offset);

  const Trace::Header& header1 = trace1.header();
  const Trace::Header& header2 = trace2.header();

  galois::gDebug(__PRETTY_FUNCTION__, "\n", header1);
  galois::gDebug(__PRETTY_FUNCTION__, "\n", header2);

  assert(kSegyHeaderXCoordCandidateOffsets.size() ==
         kSegyHeaderYCoordCandidateOffsets.size());

  std::map<Trace::Header::Attribute, int> offsets;

  for (size_t i = 0; i < kSegyHeaderXCoordCandidateOffsets.size(); i++) {
    int x_offset = kSegyHeaderXCoordCandidateOffsets[i];
    int y_offset = kSegyHeaderYCoordCandidateOffsets[i];

    float x_coord1 = header1.getCoordinateValue(x_offset);
    float y_coord1 = header1.getCoordinateValue(y_offset);
    float x_coord2 = header2.getCoordinateValue(x_offset);
    float y_coord2 = header2.getCoordinateValue(y_offset);

    if ((x_coord1 != x_coord2) || (y_coord1 != y_coord2)) {
      offsets[Trace::Header::Attribute::X_COORDINATE] = x_offset;
      offsets[Trace::Header::Attribute::Y_COORDINATE] = y_offset;
      break;
    }
  }

  assert(kSegyHeaderInlineCandidateOffsets.size() ==
         kSegyHeaderCrosslineCandidateOffsets.size());

  for (size_t i = 0; i < kSegyHeaderInlineCandidateOffsets.size(); i++) {
    int il_offset = kSegyHeaderInlineCandidateOffsets[i];
    int xl_offset = kSegyHeaderCrosslineCandidateOffsets[i];

    int32_t il1 = header1.getValueAtOffset<int32_t>(il_offset);
    int32_t xl1 = header1.getValueAtOffset<int32_t>(xl_offset);
    int32_t il2 = header2.getValueAtOffset<int32_t>(il_offset);
    int32_t xl2 = header2.getValueAtOffset<int32_t>(xl_offset);

    if ((il1 != il2) || (xl1 != xl2)) {
      offsets[Trace::Header::Attribute::INLINE_NUMBER]    = il_offset;
      offsets[Trace::Header::Attribute::CROSSLINE_NUMBER] = xl_offset;
      break;
    }
  }

  auto check_offset_exists = [&](Trace::Header::Attribute attr,
                                 const std::string& attr_name) {
    if (offsets.find(attr) == offsets.end()) {
      galois::gWarn("Warning: Could not guess the location of ", attr_name,
                    " in the trace header!");
    } else {
      galois::gDebug(attr_name, " at offset ", offsets.find(attr)->second);
    }
  };

  check_offset_exists(Trace::Header::Attribute::INLINE_NUMBER, "inline number");
  check_offset_exists(Trace::Header::Attribute::CROSSLINE_NUMBER,
                      "crossline number");
  check_offset_exists(Trace::Header::Attribute::X_COORDINATE, "X coordinate");
  check_offset_exists(Trace::Header::Attribute::Y_COORDINATE, "Y coordinate");

  return offsets;
}

void SegyFile::open(std::ios_base::openmode mode) {
  galois::gDebug(__PRETTY_FUNCTION__);
  if (file_->is_open()) {
    close();
  }

  mode_ = mode;
  file_->open(mode);
  file_->map();

  char* start_addr = file_->char_addr();
  text_header_     = reinterpret_cast<TextHeader*>(start_addr);

  galois::gDebug("Textual File Header\n", *text_header_);

  start_addr += sizeof(TextHeader);
  binary_header_ = reinterpret_cast<BinaryHeader*>(start_addr);

  num_samples_per_trc_ = binary_header_->get_field<BFHNumSamplesPerTrace>();
  num_ext_hdrs_        = binary_header_->get_field<BFHNumExtTFH>();

  galois::gDebug("Binary File Header:\n", *binary_header_);
  galois::gDebug(BFHNumSamplesPerTrace::desc, ": ", num_samples_per_trc_);
  galois::gDebug(BFHNumExtTFH::desc, ": ", num_ext_hdrs_);

  if (num_ext_hdrs_ < 0) {
    throw std::runtime_error(
        "SEGY: Cannot handle variable number of extended headers yet!");
  }

  first_hdr_ptr_ = file_->char_addr() + sizeof(TextHeader) +
                   sizeof(BinaryHeader) + num_ext_hdrs_ * sizeof(TextHeader);
  seek(0);
}

double SegyFile::Trace::Header::getCoordinateValue(int offset) const {
  double coord                = double(getValueAtOffset<int32_t>(offset));
  int16_t scalar_apply_coords = getValueAtOffset<int16_t>(71);
  if (scalar_apply_coords != 0) {
    if (scalar_apply_coords > 0) {
      coord *= double(scalar_apply_coords);
    } else {
      coord /= double(std::abs(scalar_apply_coords));
    }
  }

  return coord;
}

void SegyFile::close() {
  text_header_   = nullptr;
  binary_header_ = nullptr;
  file_->unmap();
  file_->close();
}

void SegyFile::seek(std::uint64_t offset) const {
  checkFileOpened();
  SegyFile* self = const_cast<SegyFile*>(this);

  self->hdr_ptr_ =
      first_hdr_ptr_ +
      offset * (sizeof(Trace::Header) + sizeof(float) * num_samples_per_trc_);
  self->trc_ptr_    = hdr_ptr_ + sizeof(Trace::Header);
  self->cur_offset_ = offset;
}

bool SegyFile::read(Trace& trace) const {
  checkFileOpened();

  Trace::Header& header       = trace.header();
  std::vector<float>& samples = trace.samples();

  if (hdr_ptr_ >= file_->end() ||
      (hdr_ptr_ + sizeof(Trace::Header)) >= file_->end()) {
    return false;
  }

  std::memcpy(&header, hdr_ptr_, sizeof(Trace::Header));

  size_t trace_bytes = sizeof(float) * num_samples_per_trc_;
  if (trc_ptr_ >= file_->end() || (trc_ptr_ + trace_bytes) > file_->end())
    return false;

  samples.resize(num_samples_per_trc_);
  std::memcpy(&(samples[0]), trc_ptr_, trace_bytes);

  uint16_t sample_format_code =
      binary_header_->get_field<BFHSampleFormatCode>();

  if (sample_format_code == BFHSampleFormatCode::IEEE) {
#if __BYTE_ORDER == __LITTLE_ENDIAN
    std::transform(samples.begin(), samples.end(), samples.begin(),
                   [](float a) -> float { return swap_endianness(a); });
#endif
  } else if (sample_format_code == BFHSampleFormatCode::IBM) {
    std::transform(samples.begin(), samples.end(), samples.begin(),
                   [](float a) -> float { return ibm_to_ieee(a, true); });
  } else {
    throw std::runtime_error("Segy: read: Data format not supported : " +
                             std::to_string(sample_format_code));
  }

  return true;
}

const SegyFile::TextHeader& SegyFile::getTextHeader() const {
  checkFileOpened();
  return *text_header_;
}

void SegyFile::setTextHeader(const SegyFile::TextHeader& header) {
  checkFileOpened();
  (*text_header_) = header;
}

const SegyFile::BinaryHeader& SegyFile::getBinaryHeader() const {
  checkFileOpened();
  return *binary_header_;
}

void SegyFile::setBinaryHeader(const SegyFile::BinaryHeader& header) {
  checkFileOpened();
  (*binary_header_) = header;
}

std::string SegyFile::TextHeader::toString() const {
  std::ostringstream ostr;
  bool isEBCDIC = isTextInEBCDICFormat(&data[0], sizeof(data));
  std::string line(kSegyTextHeaderLineWidth, ' ');
  for (size_t i = 0, j = 0; i < sizeof(data); i++, j++) {
    j = j % kSegyTextHeaderLineWidth;
    if (isEBCDIC) {
      line[j] = getASCIIForEBCDIC(data[i]);
    } else {
      line[j] = data[i];
    }

    if ((i + 1) % kSegyTextHeaderLineWidth == 0) {
      ostr << line << std::endl;
    }
  }

  return ostr.str();
}

void SegyFile::BinaryHeader::print(std::ostream& os) const {
  os << "SEGY Version : " << int(get_field<BFHRevMajor>()) << "."
     << (int)get_field<BFHRevMinor>() << std::endl;
  os << BFHNumTracesPerEnsemble::desc << ": "
     << (int)get_field<BFHNumTracesPerEnsemble>() << std::endl;
  os << BFHNumAuxTracesPerEnsemble::desc << ": "
     << (int)get_field<BFHNumAuxTracesPerEnsemble>() << std::endl;
  os << BFHSampleInterval::desc << ": " << (int)get_field<BFHSampleInterval>()
     << " (us)" << std::endl;
  os << BFHNumSamplesPerTrace::desc << ": "
     << (int)get_field<BFHNumSamplesPerTrace>() << std::endl;
  os << BFHSampleFormatCode::desc << ": "
     << (int)get_field<BFHSampleFormatCode>() << std::endl;
  os << BFHEnsembleFold::desc << ": " << (int)get_field<BFHEnsembleFold>()
     << std::endl;
  os << BFHTraceSortingCode::desc << ": "
     << (int)get_field<BFHTraceSortingCode>() << std::endl;
  os << BFHMsmSys::desc << ": " << (int)get_field<BFHMsmSys>() << std::endl;
  os << BFHTraceFlag::desc << ": " << (int)get_field<BFHTraceFlag>()
     << std::endl;
}

void SegyFile::Trace::Header::print(std::ostream& os) const {
  os << THSeqNumInLine::desc << ": " << (int)get_field<THSeqNumInLine>()
     << std::endl;
  os << THOrigFieldRecNum::desc << ": " << (int)get_field<THOrigFieldRecNum>()
     << std::endl;
  os << THNumInRec::desc << ": " << (int)get_field<THNumInRec>() << std::endl;
  os << THIDCode::desc << ": " << (int)get_field<THIDCode>() << std::endl;
  os << THNumSamples::desc << ": " << (int)get_field<THNumSamples>()
     << std::endl;
  os << THSampleInterval::desc << ": " << (int)get_field<THSampleInterval>()
     << std::endl;

  os << "** Elevations and Depths (Byte 41-70)" << std::endl;
  os << THElevDptScalar::desc << ": " << (int)get_field<THElevDptScalar>()
     << std::endl;
  os << THRecvGrpElev::desc << ": " << (int)get_field<THRecvGrpElev>()
     << std::endl;
  os << THSurfElevAtSrc::desc << ": " << (int)get_field<THSurfElevAtSrc>()
     << std::endl;
  os << THSrcDptBelowSurf::desc << ": " << (int)get_field<THSrcDptBelowSurf>()
     << std::endl;
  os << THDatumElevAtRecvGrp::desc << ": "
     << (int)get_field<THDatumElevAtRecvGrp>() << std::endl;
  os << THDatumElevAtSrc::desc << ": " << (int)get_field<THDatumElevAtSrc>()
     << std::endl;
  os << THWaterDptAtSrc::desc << ": " << (int)get_field<THWaterDptAtSrc>()
     << std::endl;
  os << THWaterDptAtRecvGrp::desc << ": "
     << (int)get_field<THWaterDptAtRecvGrp>() << std::endl;

  os << "** Possible inline/crossline locations: " << std::endl;
  os << "3-D poststack data: "
     << "(" << (int)get_field<TH3DPoststackInlineNum>() << ", "
     << (int)get_field<TH3DPoststackCrlineNum>() << ")" << std::endl;
  os << "Device/Trace ID & Source Type/Orientaion: "
     << "(" << (int)get_field<THDeviceID>() << ", "
     << (int)get_field<THSrcTyAndOriCode>() << ")" << std::endl;
  os << "Original field record number & Ensemble number: "
     << "(" << (int)get_field<THOrigFieldRecNum>() << ", "
     << (int)get_field<THEnsmNum>() << ")" << std::endl;
  // for (size_t i = 0; i < kSegyHeaderInlineCandidateOffsets.size(); i++) {
  //   int il_offset = kSegyHeaderInlineCandidateOffsets[i];
  //   int xl_offset = kSegyHeaderCrosslineCandidateOffsets[i];

  //   int32_t il = getValueAtOffset<int32_t>(il_offset);
  //   int32_t xl = getValueAtOffset<int32_t>(xl_offset);
  //   os << "Offset (" << il_offset << ", " << xl_offset << ") -> (" << il <<
  //   ", "
  //      << xl << ")" << std::endl;
  // }

  os << "** Possible coordinate locations: " << std::endl;
  os << "Ensemble (CDP) position: "
     << "(" << (int)get_field<THEnsmX>() << ", " << (int)get_field<THEnsmY>()
     << ")" << std::endl;
  os << "Source coordinate: "
     << "(" << (int)get_field<THSrcX>() << ", " << (int)get_field<THSrcY>()
     << ")" << std::endl;
  os << "Shotpoint number scalar & Transduction Constant: "
     << "(" << (int)get_field<THShotpointNumScalar>() << ", "
     << (int)get_field<THTransConstMantissa>() << " * 10^"
     << (int)get_field<THTransConstExponent>() << ")" << std::endl;
  // for (size_t i = 0; i < kSegyHeaderXCoordCandidateOffsets.size(); i++) {
  //   int x_offset = kSegyHeaderXCoordCandidateOffsets[i];
  //   int y_offset = kSegyHeaderYCoordCandidateOffsets[i];

  //   float x_coord = getCoordinateValue(x_offset);
  //   float y_coord = getCoordinateValue(y_offset);
  //   os << "Offset (" << x_offset << ", " << y_offset << ") -> (" << x_coord
  //      << ", " << y_coord << ")" << std::endl;
  // }

  os << THTransUnitsCode::desc << ": " << (int)get_field<THTransUnitsCode>()
     << std::endl;
}

void SegyFile::checkFileOpened() const {
  if (!file_->is_open()) {
    std::ostringstream ostr;
    ostr << "File '" << file_->name() << "' has not been opened!" << std::endl;
    throw std::runtime_error(ostr.str());
  }
}

void SegyFile::checkFileOpenedForReading() const {
  if (!file_->is_open() || mode_ & std::ios_base::in) {
    std::ostringstream ostr;
    ostr << "File '" << file_->name() << "' has not been opened!" << std::endl;
    throw std::runtime_error(ostr.str());
  }
}

} // namespace segystack
