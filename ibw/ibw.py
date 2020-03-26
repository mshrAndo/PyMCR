#!/usr/bin/python
# -*- coding: utf-8 -*-

import struct
import logging

import time

class IgorWave(object):
    # wave_type 
    NT_TEXT     = 0x00 #
    NT_FP32    = 0x02 # single float
    NT_FP64    = 0x04 # double flost

    NT_I8      = 0x08 # int8_t
    NT_I16     = 0x10 # int16_t
    NT_I32     = 0x20 # int32_t
    
    NT_FLAG_COMPLEX  = 0x01 # 'complex' flag
    NT_FLAG_UNSIGNED = 0x40 # 'unsigned'

    MAX_DIMS = 4

    def __init__(self, path = None):
        if not path is None:
            self.note =''
            self.load_file(path)

    def validate(self):
        #logging.error(["should 1", self.wave_header['whVersion']])
        if self.bin_header['version'] == 2:
            return True

        if self.bin_header['version'] == 5:
            return self.wave_header['whVersion'] == 1

        return False

    def load_file(self, src):
        """
        >>> i = IgorWave()
        >>> i.load_file(open('./matrix_for_mp_20110427_01.ibw'))
        >>> len(i.blob)
        1638400
        """        
        src.seek(0)
        bin_version = struct.unpack("H", src.read(2))[0]
        src.seek(0)

        if bin_version == 2:
            self.bin_header = IgorBinV2Header(src)
            self.wave_header = IgorWaveV2Header(src)
            logging.error(['tell:', src.tell()])
            self.blob = src.read(self.bin_header['wfmSize'] - 126)
            logging.info([self.bin_header])
            logging.info([self.wave_header])
            logging.info(["data len", self.bin_header['wfmSize'] - 126,
                          len(self.blob)])

        elif bin_version == 5:
            self.bin_header = IgorBinV5Header(src)
            self.wave_header = IgorWaveV5Header(src)
            self.note = ''
            self.blob = src.read(self.bin_header['wfmSize'] - 320)
            src.seek(64+self.bin_header['wfmSize']+self.bin_header['formulaSize'])
            if self.bin_header['noteSize'] > 0:
                 self.note = src.read(self.bin_header['noteSize'])
            
            logging.info([self.bin_header])
            logging.info([self.wave_header])
            logging.info(["data len", self.bin_header['wfmSize'] - 320,
                          len(self.blob)])
            logging.info([self.note])

    def set_blob(self, name, size, blob, data_type = None, note = ''):
        if data_type is None:
            data_type = self.NT_FP32
        self.bin_header = IgorBinV5Header(
            fields = {
                'checksum': 0,
                'dataEUnitsSize': 0,
                'dimEUnitsSize': [0, 0, 0, 0],
                'dimLabelsSize': [0, 0, 0, 0],
                'formulaSize': 0,
                'noteSize': len(note),
                'sIndicesSize': 0,
                'version': 5,
                'wfmSize': len(blob) + 320,
                })
        self.wave_header = IgorWaveV5Header(
            fields = {
                'bname': name,
                'botFullScale': 0.0,
                'creationDate': 2082844800+(int(time.mktime(time.localtime()))-time.timezone),
                'fsValid': 0,
                'modDate': 2082844800+(int(time.mktime(time.localtime()))-time.timezone),
                'nDim': [size[0], size[1], size[2], 0],
                'npnts': max((size[0] * size[1] * size[2]),max((size[0] * size[1]), size[0])),
                'platform': 2,
                'sfA': [1.0, 1.0, 1.0, 1.0],
                'sfB': [0.0, 0.0, 0.0, 0.0],
                'topFullScale': 0.0,
                'type': data_type,
                'whVersion': 1,
                })
        self.blob = blob
        self.note = note

    def to_numpy_array(self, numpy):
        if   self.wave_header['type'] == self.NT_I8:
            A = numpy.frombuffer(self.blob, dtype = numpy.int8)
        elif self.wave_header['type'] == self.NT_I8 + self.NT_FLAG_UNSIGNED:
            A = numpy.frombuffer(self.blob, dtype = numpy.uint8)

        elif self.wave_header['type'] == self.NT_I16:
            A = numpy.frombuffer(self.blob, dtype = numpy.int16)
        elif self.wave_header['type'] == self.NT_I16 + self.NT_FLAG_UNSIGNED:
            A = numpy.frombuffer(self.blob, dtype = numpy.uint16)

        elif self.wave_header['type'] == self.NT_I32:
            A = numpy.frombuffer(self.blob, dtype = numpy.int32)
        elif self.wave_header['type'] == self.NT_I32 + self.NT_FLAG_UNSIGNED:
            A = numpy.frombuffer(self.blob, dtype = numpy.uint32)

        elif self.wave_header['type'] == self.NT_FP32:
            A = numpy.frombuffer(self.blob, dtype = numpy.float32)
        elif self.wave_header['type'] == self.NT_FP64:
            A = numpy.frombuffer(self.blob, dtype = numpy.float64)
        else:
            raise ValueError(['invalid type', self.wave_header['type']])
        if self.bin_header['version'] == 2:
            return A # 1D array
        else:
            rows, cols = self.wave_header['nDim'][:2]
            logging.info([A.shape, 'as matrix:', (rows, cols)])
            if cols > 0:
                return numpy.reshape(A, (rows, cols), 'F') # to 2D array
            else:
                return numpy.reshape(A, (rows, 1), 'F') # to pseudo 2D array

    def save_file(self, dest):
        """
        >>> i = IgorWave()
        >>> i.load_file(open('./matrix_for_mp_20110427_01.ibw'))
        >>> i.save_file(open('./saved.ibw', 'w'))
        >>> i2 = IgorWave()
        >>> i2.load_file(open('./saved.ibw'))
        >>> repr(i.bin_header) == repr(i2.bin_header)
        True
        >>> repr(i.wave_header) == repr(i2.wave_header)
        True
        """
        checksum_old = self.bin_header['checksum']
        self.bin_header['checksum'] = 0
        blob_b = self.bin_header.to_blob()
        blob_w = self.wave_header.to_blob()

        # sum first 384 bytes as int16
        checksum_new = (sum(struct.unpack("192h", blob_b + blob_w)) % 65536)
        #print checksum_old, checksum_new

        self.bin_header['checksum'] = 65536 - checksum_new
        blob_b = self.bin_header.to_blob()

        dest.write(blob_b)
        dest.write(blob_w)
        dest.write(self.blob)
        dest.write(self.note)
        
class _IgorHeader(object):
    def __init__(self, src = None, fields = None):
        self.blob = ""
        if src:
            self.decode_blob(src)
        elif fields:
            self.fields = dict(fields)
        
    def decode_blob(self, src):
        """decode raw data"""
        if not src:
            return
        fmt_str = self.fmt_str()
        blob = src.read(struct.calcsize(fmt_str))
        values = list(struct.unpack(fmt_str, blob))

        self.fields = {}
        for desc in self.FORMAT:
            if not desc[1]: # pads
                continue
            elif len(desc) == 3: # array
                code, name, rep = desc
                self.fields[name] = values[:rep]
                values = values[rep:]
            else:
                code, name = desc
                self.fields[name] = values.pop(0)

    def to_blob(self):
        values = []
        for desc in self.FORMAT:
            if not desc[1]: # pads
                continue
            elif len(desc) == 3:# array
                code, name, rep = desc
                if len(self.fields[name]) != rep:
                    raise ValueError(desc, self.fields[name])
                values.extend(self.fields[name])
            else:
                code, name = desc
                values.append(self.fields[name])
        return struct.pack(self.fmt_str(), *values)

    def fmt_str(self):
        """format string to be used with the struct module"""
        # cancel padding by '='
        return '=' + "".join( ( (("%d" % pair[2]) if len(pair) == 3 else "")
                                + pair[0]) for pair in self.FORMAT)
    def __getitem__(self, key):
        return self.fields[key]
    def __setitem__(self, key, value):
        self.fields[key] = value
    def __repr__(self):
        return repr([(k, self[k]) for k in sorted(self.fields)])

class IgorBinV5Header(_IgorHeader):
    """
    >>> i = IgorBinV5Header(open('./matrix_for_mp_20110427_01.ibw'))
    >>> repr(i)
    "[('checksum', 36907), ('dataEUnitsSize', 0), ('dimEUnitsSize', [0, 0, 0, 0]), ('dimLabelsSize', [0, 0, 0, 0]), ('formulaSize', 0), ('noteSize', 0), ('sIndicesSize', 0), ('version', 5), ('wfmSize', 1638720)]"

    >>> fmt_str = IgorBinV5Header(None).fmt_str()
    >>> fmt_str
    '=HHIIII4I4II8x'
    >>> struct.calcsize(fmt_str)
    64
    """
    FORMAT = [
        ("H", "version"), #  should be 5
        ("H", "checksum"), #  sum(first 384 bytes as int16[]) should be 0
        ("I", "wfmSize"), #  WaveHeader5 + wave data.
        ("I", "formulaSize"),
        ("I", "noteSize"),
        ("I", "dataEUnitsSize"), #  size of extended data units.
        ("I", "dimEUnitsSize", IgorWave.MAX_DIMS),
        ("I", "dimLabelsSize", IgorWave.MAX_DIMS),
        ("I", "sIndicesSize"), #  size of string indicies (text wave only)
        ("x", None, 8) #  reserved
        ]

class IgorWaveV5Header(_IgorHeader):
    """
    >>> fmt_str = IgorWaveV5Header(None).fmt_str()
    >>> fmt_str
    '=4xIIih2x6xh32s4x4x4i4d4d4x16xh2xdd4x16x16x4xB3x52x4x4x2x2x2xxx4x4x2x2x4x4x'
    >>> struct.calcsize(fmt_str)
    320

    >>> f = open('./matrix_for_mp_20110427_01.ibw')
    >>> bin_header = IgorBinV5Header(f)
    >>> i = IgorWaveV5Header(f)
    >>> del i.fields['bname']
    >>> [(k, i[k]) for k in sorted(i.fields)]
    [('botFullScale', 0.0), ('creationDate', 3386870125L), ('fsValid', 0), ('modDate', 3386870125L), ('nDim', [400, 1024, 0, 0]), ('npnts', 409600), ('platform', 2), ('sfA', [1.0, 1.0, 1.0, 1.0]), ('sfB', [0.0, 0.0, 0.0, 0.0]), ('topFullScale', 0.0), ('type', 2), ('whVersion', 1)]
    """
    MAX_UNIT_CHARS = 3
    MAX_WAVE_NAME_5 = 31
    FORMAT = [
        ("x", None, 4),
        ("I", "creationDate"), # ctime
        ("I", "modDate"), # mtime
        ("i", "npnts"), # total number of points in this wave
        ("h", "type"), # enum wave_type
        ("x", None, 2), # dLock
        ("x", None, 6), # wgpad
        ("h", "whVersion"), # write 1 for version 5
        ("%ds" % (MAX_WAVE_NAME_5+1), "bname"), # name of wave
        ("x", None, 4), # whpad2
        ("x", None, 4), # dFolder
         # wave dimension
        ("i", "nDim", IgorWave.MAX_DIMS), # size in each dimension
        ("d", "sfA",  IgorWave.MAX_DIMS),
        ("d", "sfB",  IgorWave.MAX_DIMS),
        # units
        ("x", None, (MAX_UNIT_CHARS + 1)), # (dataUnits)
        ("x", None, (IgorWave.MAX_DIMS * (MAX_UNIT_CHARS + 1)) ), # (dataUnits)
        ("h", "fsValid"), # 1 if top/bot full scale values are valid
        ("x", None, 2), # whPad3
        ("d", "topFullScale"),
        ("d", "botFullScale"),
        ("x", None, 4), # (dataEUnits)
        ("x", None, (IgorWave.MAX_DIMS * 4)), # (dimEUnits)
        ("x", None, (IgorWave.MAX_DIMS * 4)), # (dimLabels)
        ("x", None, 4), # (waveNoteH)
        ("B", "platform"), # 0: unspec. 1: Mac, 2:Windows
        ("x", None, 3), #spare
        ("x", None, (4 * 13)), # whUnused
        ("4x", None), # (vRefNum)
        ("4x", None), # (dirID)
        ("x", None, 2), # (aModified)
        ("x", None, 2), # (wModified)
        ("x", None, 2), # (swModified)
        ("x", None), # (useBits)
        ("x", None), # (kindBits)
        ("x", None, 4), # (formula)
        ("x", None, 4), # (depID)
        ("x", None, 2), # (whpad4)
        ("x", None, 2), # (srcFldr)
        ("x", None, 4), # (fileName)
        ("x", None, 4), # (sIndices)
        ] # 320 bytes


class IgorBinV2Header(_IgorHeader):
    """
    """
    FORMAT = [
        ("H", "version"), #  should be 5
        ("I", "wfmSize"), #  WaveHeader5 + wave data.
        ("I", "noteSize"),
        ("I", "pictSize"),
        ("H", "checksum"),
        ]

class IgorWaveV2Header(_IgorHeader):
    """
    """
    MAX_UNIT_CHARS = 3
    MAX_WAVE_NAME_2 = 18
    FORMAT = [
        ("h", "type"), # enum wave_type
        ("x", None, 4), # next
        ("%ds" % (MAX_WAVE_NAME_2+2), "bname"), # name of wave
        ("h", "whVersion"),
        ("x", None, 2), # src folder
        ("x", None, 4), # file name
        ("x", None, 4), # data units
        ("x", None, 4), # x units
        ("I", "npnts"),
        ("x", None, 2), # aModified
        ("d", "hsA"),
        ("d", "hsB"),
        ("x", None, 2), # wModified
        ("x", None, 2), # swModified
        ("h", "fsValid"), # full scales have meaning
        ("d", "topFullScale"),
        ("d", "botFullScale"),
        ("x", None, 2), # useBits, kindBits
        ("x", None, 4), # formula
        ("x", None, 4), # depID
        ("I", "creationDate"),
        ("x", None, 2), # wUnused
        ("I", "modDate"),
        ("x", None, 4), # waveNoteH
        ] # 


if __name__== "__main__":
    import doctest
    doctest.testmod()